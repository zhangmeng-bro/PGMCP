import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
windowSize = 25
K = 30  # 参考Hybrid-Spectral-Net
rate = 16


class HybridSN(nn.Module):
    # 定义各个层的部分
    def __init__(self):
        super(HybridSN, self).__init__()
        self.S = windowSize
        self.L = K

        # self.conv_block = nn.Sequential()
        ## convolutional layers
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(7, 3, 3))
        self.conv2 = nn.Conv3d(in_channels=8, out_channels=16, kernel_size=(5, 3, 3))
        self.conv3 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(3, 3, 3))

        # 不懂 inputX经过三重3d卷积的大小
        inputX = self.get2Dinput()
        inputConv4 = inputX.shape[1] * inputX.shape[2]
        # conv4 （24*24=576, 19, 19），64个 3x3 的卷积核 ==>（（64, 17, 17）
        self.conv4 = nn.Conv2d(inputConv4, 64, kernel_size=(3, 3))

        # self-attention
        self.sa1 = nn.Conv2d(64, 64 // rate, kernel_size=1)
        self.sa2 = nn.Conv2d(64 // rate, 64, kernel_size=1)

        # 全连接层（256个节点） # 64 * 17 * 17 = 18496
        self.dense1 = nn.Linear(18496, 256)
        # 全连接层（128个节点）
        self.dense2 = nn.Linear(256, 128)
        # 最终输出层(16个节点)
        # self.dense3 = nn.Linear(128, class_num)

        # 让某个神经元的激活值以一定的概率p，让其停止工作，这次训练过程中不更新权值，也不参加神经网络的计算。
        # 但是它的权重得保留下来（只是暂时不更新而已），因为下次样本输入时它可能又得工作了
        # 参考: https://blog.csdn.net/yangfengling1023/article/details/82911306
        # self.drop = nn.Dropout(p = 0.4)
        # 改成0.43试试
        self.drop = nn.Dropout(p=0.43)
        self.soft = nn.Softmax(dim=1)
        pass

    # 辅助函数，没怎么懂，求经历过三重卷积后二维的一个大小
    def get2Dinput(self):
        # torch.no_grad(): 做运算，但不计入梯度记录
        with torch.no_grad():
            x = torch.zeros((1, 1, self.L, self.S, self.S))
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
        return x
        pass

    # 必须重载的部分，X代表输入
    def forward(self, x):
        # F在上文有定义torch.nn.functional，是已定义好的一组名称
        out = self.conv1(x)
        out = F.relu(out)
        out = self.conv2(out)
        out = F.relu(out)
        out = self.conv3(out)
        out = F.relu(out)

        # 进行二维卷积，因此把前面的 32*18 reshape 一下，得到 （576, 19, 19）
        out = out.view(-1, out.shape[1] * out.shape[2], out.shape[3], out.shape[4])
        out = self.conv4(out)
        out = F.relu(out)

        weight = F.avg_pool2d(out, out.size(2))  # 参数为输入，kernel

        weight = F.relu(self.sa1(weight))
        weight = F.sigmoid(self.sa2(weight))
        out = out * weight

        # flatten: 变为 18496 维的向量，
        out = out.view(out.size(0), -1)

        out = F.relu(self.dense1(out))
        out = self.drop(out)
        out = F.relu(self.dense2(out))
        out = self.drop(out)
        out = self.dense3(out)

        return out


class SSSE(nn.Module):

    def __init__(self, input_channels, patch_size, n_classes):
        super(SSSE, self).__init__()
        self.kernel_dim = 1
        self.feature_dim = input_channels
        self.sz = patch_size
        # Convolution Layer 1 kernel_size = (1, 1, 7), stride = (1, 1, 2), output channels = 24
        self.conv1 = nn.Conv3d(1, 24, kernel_size=(7, 1, 1), stride=(2, 1, 1), bias=True)
        self.bn1 = nn.BatchNorm3d(24)
        self.activation1 = nn.ReLU()

        # Residual block 1
        self.conv2 = nn.Conv3d(24, 24, kernel_size=(7, 1, 1), stride=1, padding=(3, 0, 0), padding_mode='replicate',
                               bias=True)
        self.bn2 = nn.BatchNorm3d(24)
        self.activation2 = nn.ReLU()
        self.conv3 = nn.Conv3d(24, 24, kernel_size=(7, 1, 1), stride=1, padding=(3, 0, 0), padding_mode='replicate',
                               bias=True)
        self.bn3 = nn.BatchNorm3d(24)
        self.activation3 = nn.ReLU()
        # Finish

        # Convolution Layer 2 kernel_size = (1, 1, (self.feature_dim - 6) // 2), output channels = 128
        self.conv4 = nn.Conv3d(24, 128, kernel_size=(((self.feature_dim - 7) // 2 + 1), 1, 1), bias=True)
        self.bn4 = nn.BatchNorm3d(128)
        self.activation4 = nn.ReLU()
        if patch_size == 7:
            self.SpectralSE = SpectralSE(128, 128, self.sz)
        elif patch_size == 3:
            self.SpectralSE = SpectralSE_three(128, 128, self.sz)
        elif patch_size == 5:
            self.SpectralSE = SpectralSE_five(128, 128, self.sz)
        elif patch_size == 9:
            self.SpectralSE = SpectralSE_nine(128,128,self.sz)
        elif patch_size == 11:
            self.SpectralSE = SpectralSE_eleven(128, 128, self.sz)
        else:
            raise Exception('this patch not define')
        # self.SpectralSE = SpectralSE_R(128, 128, self.sz)
        # self.SpectralSE = SpectralSE_S(128, 128, self.sz)

        # Convolution layer for spatial information
        self.conv5 = nn.Conv3d(1, 24, (self.feature_dim, 1, 1))
        self.bn5 = nn.BatchNorm3d(24)
        self.activation5 = nn.ReLU()

        # Residual block 2
        self.conv6 = nn.Conv3d(24, 24, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), padding_mode='replicate',
                               bias=True)
        self.bn6 = nn.BatchNorm3d(24)
        self.activation6 = nn.ReLU()
        self.conv7 = nn.Conv3d(24, 24, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), padding_mode='replicate',
                               bias=True)
        self.bn7 = nn.BatchNorm3d(24)
        self.activation7 = nn.ReLU()
        self.SpatialSE = SpatialSE(24, 1)
        self.conv8 = nn.Conv3d(24, 24, kernel_size=1)
        # Finish

        # Combination shape
        self.inter_size = 128 + 24

        # Residual block 3
        self.conv9 = nn.Conv3d(self.inter_size, self.inter_size, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1),
                               padding_mode='replicate', bias=True)
        self.bn9 = nn.BatchNorm3d(self.inter_size)
        self.activation9 = nn.ReLU()
        self.conv10 = nn.Conv3d(self.inter_size, self.inter_size, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1),
                                padding_mode='replicate', bias=True)
        self.bn10 = nn.BatchNorm3d(self.inter_size)
        self.activation10 = nn.ReLU()

        # Average pooling kernel_size = (5, 5, 1)
        self.avgpool = nn.AvgPool3d((1, self.sz, self.sz))

        # Fully connected Layer
        self.fc1 = nn.Linear(in_features=self.inter_size, out_features=n_classes)

        # parameters initialization
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, bounds=None):
        # Convolution layer 1
        x = x.unsqueeze(1)
        x1 = self.conv1(x)
        x1 = self.activation1(self.bn1(x1))
        # Residual layer 1
        residual = x1
        x1 = self.conv2(x1)
        x1 = self.activation2(self.bn2(x1))
        x1 = self.conv3(x1)
        x1 = residual + x1
        x1 = self.activation3(self.bn3(x1))

        # Convolution layer to combine rest
        x1 = self.conv4(x1)
        x1 = self.activation4(self.bn4(x1))
        x1 = x1.reshape(x1.size(0), x1.size(1), x1.size(3), x1.size(4))
        e1 = self.SpectralSE(x1)
        x1 = torch.mul(e1, x1)

        x2 = self.conv5(x)
        x2 = self.activation5(self.bn5(x2))

        # Residual layer 2
        residual = x2
        residual = self.conv8(residual)
        x2 = self.conv6(x2)
        x2 = self.activation6(self.bn6(x2))
        x2 = self.conv7(x2)
        x2 = residual + x2

        x2 = self.activation7(self.bn7(x2))
        x2 = x2.reshape(x2.size(0), x2.size(1), x2.size(3), x2.size(4))
        e2 = self.SpatialSE(x2)
        x2 = torch.mul(e2,x2)

        # concat spatial and spectral information
        x = torch.cat((x1, x2), 1)

        x = self.avgpool(x)
        x = x.reshape((x.size(0), -1))

        # Fully connected layer
        # x = self.fc1(x)

        return x

class DCRN_02F(nn.Module):

    def __init__(self, input_channels, patch_size, n_classes):
        super(DCRN_02F, self).__init__()
        self.inter_size = 192 + 96
        # self.SelfNorm = SelfNorm(self.inter_size)
        # self.CrossNorm = CrossNorm()
        self.kernel_dim = 1
        self.feature_dim = input_channels
        self.sz = patch_size
        # Convolution Layer 1 kernel_size = (1, 1, 7), stride = (1, 1, 2), output channels = 24
        self.conv1 = nn.Conv3d(1, 24, kernel_size=(7, 1, 1), stride=(2, 1, 1), bias=True)
        self.bn1 = nn.BatchNorm3d(24)
        self.activation1 = nn.ReLU()

        # Residual block 1
        self.conv2 = nn.Conv3d(24, 24, kernel_size=(7, 1, 1), stride=1, padding=(3, 0, 0),
                               bias=True)  # padding_mode='replicate',
        self.bn2 = nn.BatchNorm3d(24)
        self.activation2 = nn.ReLU()
        self.conv3 = nn.Conv3d(24, 24, kernel_size=(7, 1, 1), stride=1, padding=(3, 0, 0),
                               bias=True)  # padding_mode='replicate',
        self.bn3 = nn.BatchNorm3d(24)
        self.activation3 = nn.ReLU()
        # Finish

        # Convolution Layer 2 kernel_size = (1, 1, (self.feature_dim - 6) // 2), output channels = 128
        self.conv4 = nn.Conv3d(24, 192, kernel_size=(((self.feature_dim - 7) // 2 + 1), 1, 1), bias=True)
        self.bn4 = nn.BatchNorm3d(192)
        self.activation4 = nn.ReLU()

        # Convolution layer for spatial information
        self.conv5 = nn.Conv3d(1, 24, (self.feature_dim, 1, 1))
        self.bn5 = nn.BatchNorm3d(24)
        self.activation5 = nn.ReLU()

        # Residual block 2
        self.conv6 = nn.Conv3d(24, 24, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1),
                               bias=True)  # padding_mode='replicate',
        self.bn6 = nn.BatchNorm3d(24)
        self.activation6 = nn.ReLU()
        self.conv7 = nn.Conv3d(24, 96, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1),
                               bias=True)  # padding_mode='replicate',
        self.bn7 = nn.BatchNorm3d(96)
        self.activation7 = nn.ReLU()
        self.conv8 = nn.Conv3d(24, 96, kernel_size=1)
        # Finish

        # Combination shape
        # self.inter_size = 128 + 24


        # Residual block 3
        self.conv9 = nn.Conv3d(self.inter_size, self.inter_size, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1),
                               bias=True)  # padding_mode='replicate',
        self.bn9 = nn.BatchNorm3d(self.inter_size)
        self.activation9 = nn.ReLU()
        self.conv10 = nn.Conv3d(self.inter_size, self.inter_size, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1),
                                bias=True)  # padding_mode='replicate',
        self.bn10 = nn.BatchNorm3d(self.inter_size)
        self.activation10 = nn.ReLU()

        # Average pooling kernel_size = (5, 5, 1)
        self.avgpool = nn.AvgPool3d((1, self.sz, self.sz))

        # Fully connected Layer
        self.fc1 = nn.Linear(in_features=self.inter_size, out_features=n_classes)

        # parameters initialization
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = x.contiguous()
        x = torch.angle(torch.fft.fftn(x, dim=(2, 3)))
        x = x.unsqueeze(1)  # (64,1,100,9,9)
        # Convolution layer 1
        x1 = self.conv1(x)
        x1 = self.activation1(self.bn1(x1))
        # Residual layer 1
        residual = x1
        x1 = self.conv2(x1)
        x1 = self.activation2(self.bn2(x1))
        x1 = self.conv3(x1)
        x1 = residual + x1  # (32,24,21,7,7)
        x1 = self.activation3(self.bn3(x1))

        # Convolution layer to combine rest
        x1 = self.conv4(x1)  # (32,128,1,7,7)
        x1 = self.activation4(self.bn4(x1))
        x1 = x1.reshape(x1.size(0), x1.size(1), x1.size(3), x1.size(4))  # (32,128,7,7)

        x2 = self.conv5(x)  # (32,24,1,7,7)
        x2 = self.activation5(self.bn5(x2))

        # Residual layer 2
        residual = x2
        residual = self.conv8(residual)  # (32,24,1,7,7)
        x2 = self.conv6(x2)  # (32,24,1,7,7)
        x2 = self.activation6(self.bn6(x2))
        x2 = self.conv7(x2)  # (32,24,1,7,7)
        x2 = residual + x2

        x2 = self.activation7(self.bn7(x2))
        x2 = x2.reshape(x2.size(0), x2.size(1), x2.size(3), x2.size(4))  # (32,24,7,7)

        # concat spatial and spectral information
        x = torch.cat((x1, x2), 1)  # (32,152,7,7)

        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)  # (288)
        xp = self.fc1(x)


        return xp, x

class SSSE_test(nn.Module):

    def __init__(self, input_channels, patch_size, n_classes):
        super(SSSE_test, self).__init__()
        self.kernel_dim = 1
        self.feature_dim = input_channels
        self.sz = patch_size
        # Convolution Layer 1 kernel_size = (1, 1, 7), stride = (1, 1, 2), output channels = 24
        self.conv1 = nn.Conv3d(1, 24, kernel_size=(7, 1, 1), stride=(2, 1, 1), bias=True)
        self.bn1 = nn.BatchNorm3d(24)
        self.activation1 = nn.ReLU()

        # Residual block 1 残差块
        self.conv2 = nn.Conv3d(24, 24, kernel_size=(7, 1, 1), stride=1, padding=(3, 0, 0), padding_mode='zeros',
                               bias=True)
        self.bn2 = nn.BatchNorm3d(24)
        self.activation2 = nn.ReLU()
        self.conv3 = nn.Conv3d(24, 24, kernel_size=(7, 1, 1), stride=1, padding=(3, 0, 0), padding_mode='zeros',
                               bias=True)
        self.bn3 = nn.BatchNorm3d(24)
        self.activation3 = nn.ReLU()
        # Finish

        # Convolution Layer 2 kernel_size = (1, 1, (self.feature_dim - 6) // 2), output channels = 128
        self.conv4 = nn.Conv3d(24, 128, kernel_size=(((self.feature_dim - 7) // 2 + 1), 1, 1), bias=True)
        self.bn4 = nn.BatchNorm3d(128)
        self.activation4 = nn.ReLU()
        if patch_size == 7:
            self.SpectralSE = SpectralSE(128, 128, self.sz)
        elif patch_size == 3:
            self.SpectralSE = SpectralSE_three(128, 128, self.sz)
        elif patch_size == 5:
            self.SpectralSE = SpectralSE_five(128, 128, self.sz)
        elif patch_size == 9:
            self.SpectralSE = SpectralSE_nine(128,128,self.sz)
        elif patch_size == 11:
            self.SpectralSE = SpectralSE_eleven(128, 128, self.sz)
        elif patch_size == 13:
            self.SpectralSE = SpectralSE_thirteen(128, 128, self.sz)
        elif patch_size == 1:
            self.SpectralSE = SpectralSE_one(128, 128, self.sz)
        else:
            raise Exception('this patch not define')
        # self.SpectralSE = SpectralSE_R(128, 128, self.sz)
        # self.SpectralSE = SpectralSE_S(128, 128, self.sz)

        # Convolution layer for spatial information
        self.conv5 = nn.Conv3d(1, 24, (self.feature_dim, 1, 1))
        self.bn5 = nn.BatchNorm3d(24)
        self.activation5 = nn.ReLU()

        # Residual block 2
        self.conv6 = nn.Conv3d(24, 24, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), padding_mode='replicate',
                               bias=True)
        self.bn6 = nn.BatchNorm3d(24)
        self.activation6 = nn.ReLU()
        self.conv7 = nn.Conv3d(24, 24, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), padding_mode='replicate',
                               bias=True)
        self.bn7 = nn.BatchNorm3d(24)
        self.activation7 = nn.ReLU()
        self.SpatialSE = SpatialSE(24, 1)
        self.conv8 = nn.Conv3d(24, 24, kernel_size=1)
        # Finish

        # Combination shape
        self.inter_size = 128 + 24

        # Residual block 3
        self.conv9 = nn.Conv3d(self.inter_size, self.inter_size, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1),
                               padding_mode='replicate', bias=True)
        self.bn9 = nn.BatchNorm3d(self.inter_size)
        self.activation9 = nn.ReLU()
        self.conv10 = nn.Conv3d(self.inter_size, self.inter_size, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1),
                                padding_mode='replicate', bias=True)
        self.bn10 = nn.BatchNorm3d(self.inter_size)
        self.activation10 = nn.ReLU()

        # Average pooling kernel_size = (5, 5, 1)
        self.avgpool = nn.AvgPool3d((1, self.sz, self.sz))
        # self.avgpool = nn.Conv2d(in_channels=self.inter_size,out_channels=self.inter_size,kernel_size=self.sz)

        # Fully connected Layer
        self.fc1 = nn.Linear(in_features=self.inter_size, out_features=n_classes)

        # parameters initialization
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, bounds=None):
        #x=(32,102,9,9)
        # Convolution layer 1
        x = x.unsqueeze(1)
        # x=(32,1,102,9,9)
        # spatial
        x1 = self.conv1(x)
        # x1=(32,24,48,9,9)
        x1 = self.activation1(self.bn1(x1))
        # x1=(32,24,48,9,9)
        # Residual layer 1
        residual = x1
        x1 = self.conv2(x1)
        x1 = self.activation2(self.bn2(x1))
        x1 = self.conv3(x1)
        x1 = residual + x1
        x1 = self.activation3(self.bn3(x1))
        # x1=(32,128,1,9,9)
        # Convolution layer to combine rest
        x1 = self.conv4(x1)
        # x1=(32,128,9,9)
        x1 = self.activation4(self.bn4(x1))
        x1 = x1.reshape(x1.size(0), x1.size(1), x1.size(3), x1.size(4))
        e1 = self.SpectralSE(x1)
        x1 = torch.mul(e1, x1)

        # spatial
        x2 = self.conv5(x)
        x2 = self.activation5(self.bn5(x2))

        # Residual layer 2
        residual = x2
        residual = self.conv8(residual)
        x2 = self.conv6(x2)
        x2 = self.activation6(self.bn6(x2))
        x2 = self.conv7(x2)
        x2 = residual + x2

        x2 = self.activation7(self.bn7(x2))
        x2 = x2.reshape(x2.size(0), x2.size(1), x2.size(3), x2.size(4))
        e2 = self.SpatialSE(x2)
        x2 = torch.mul(e2,x2)

        # concat spatial and spectral information
        x = torch.cat((x1, x2), 1)

        x = self.avgpool(x)
        x = x.reshape((x.size(0), -1))

        # Fully connected layer
        # x = self.fc1(x)

        return x


class SSSE_testfft(nn.Module):

    def __init__(self, input_channels, patch_size, n_classes):
        super(SSSE_testfft, self).__init__()
        self.kernel_dim = 1
        self.feature_dim = input_channels
        self.sz = patch_size
        # Convolution Layer 1 kernel_size = (1, 1, 7), stride = (1, 1, 2), output channels = 24
        self.conv1 = nn.Conv3d(1, 24, kernel_size=(7, 1, 1), stride=(2, 1, 1), bias=True)
        self.bn1 = nn.BatchNorm3d(24)
        self.activation1 = nn.ReLU()

        # Residual block 1 残差块
        self.conv2 = nn.Conv3d(24, 24, kernel_size=(7, 1, 1), stride=1, padding=(3, 0, 0), padding_mode='zeros',
                               bias=True)
        self.bn2 = nn.BatchNorm3d(24)
        self.activation2 = nn.ReLU()
        self.conv3 = nn.Conv3d(24, 24, kernel_size=(7, 1, 1), stride=1, padding=(3, 0, 0), padding_mode='zeros',
                               bias=True)
        self.bn3 = nn.BatchNorm3d(24)
        self.activation3 = nn.ReLU()
        # Finish

        # Convolution Layer 2 kernel_size = (1, 1, (self.feature_dim - 6) // 2), output channels = 128
        self.conv4 = nn.Conv3d(24, 128, kernel_size=(((self.feature_dim - 7) // 2 + 1), 1, 1), bias=True)
        self.bn4 = nn.BatchNorm3d(128)
        self.activation4 = nn.ReLU()
        if patch_size == 7:
            self.SpectralSE = SpectralSE(128, 128, self.sz)
        elif patch_size == 3:
            self.SpectralSE = SpectralSE_three(128, 128, self.sz)
        elif patch_size == 5:
            self.SpectralSE = SpectralSE_five(128, 128, self.sz)
        elif patch_size == 9:
            self.SpectralSE = SpectralSE_nine(128,128,self.sz)
        elif patch_size == 11:
            self.SpectralSE = SpectralSE_eleven(128, 128, self.sz)
        else:
            raise Exception('this patch not define')
        # self.SpectralSE = SpectralSE_R(128, 128, self.sz)
        # self.SpectralSE = SpectralSE_S(128, 128, self.sz)

        # Convolution layer for spatial information
        self.conv5 = nn.Conv3d(1, 24, (self.feature_dim, 1, 1))
        self.bn5 = nn.BatchNorm3d(24)
        self.activation5 = nn.ReLU()

        # Residual block 2
        self.conv6 = nn.Conv3d(24, 24, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), padding_mode='replicate',
                               bias=True)
        self.bn6 = nn.BatchNorm3d(24)
        self.activation6 = nn.ReLU()
        self.conv7 = nn.Conv3d(24, 24, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), padding_mode='replicate',
                               bias=True)
        self.bn7 = nn.BatchNorm3d(24)
        self.activation7 = nn.ReLU()
        self.SpatialSE = SpatialSE(24, 1)
        self.conv8 = nn.Conv3d(24, 24, kernel_size=1)
        # Finish

        # Combination shape
        self.inter_size = 128 + 24

        # Residual block 3
        self.conv9 = nn.Conv3d(self.inter_size, self.inter_size, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1),
                               padding_mode='replicate', bias=True)
        self.bn9 = nn.BatchNorm3d(self.inter_size)
        self.activation9 = nn.ReLU()
        self.conv10 = nn.Conv3d(self.inter_size, self.inter_size, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1),
                                padding_mode='replicate', bias=True)
        self.bn10 = nn.BatchNorm3d(self.inter_size)
        self.activation10 = nn.ReLU()

        # Average pooling kernel_size = (5, 5, 1)
        self.avgpool = nn.AvgPool3d((1, self.sz, self.sz))
        # self.avgpool = nn.Conv2d(in_channels=self.inter_size,out_channels=self.inter_size,kernel_size=self.sz)

        # Fully connected Layer
        self.fc1 = nn.Linear(in_features=self.inter_size, out_features=n_classes)

        # parameters initialization
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, bounds=None):
        x = torch.angle(torch.fft.fftn(x, dim=(2, 3)))
        #x=(32,102,9,9)
        # Convolution layer 1
        x = x.unsqueeze(1)
        # x=(32,1,102,9,9)
        # spatial
        x1 = self.conv1(x)
        # x1=(32,24,48,9,9)
        x1 = self.activation1(self.bn1(x1))
        # x1=(32,24,48,9,9)
        # Residual layer 1
        residual = x1
        x1 = self.conv2(x1)
        x1 = self.activation2(self.bn2(x1))
        x1 = self.conv3(x1)
        x1 = residual + x1
        x1 = self.activation3(self.bn3(x1))
        # x1=(32,128,1,9,9)
        # Convolution layer to combine rest
        x1 = self.conv4(x1)
        # x1=(32,128,9,9)
        x1 = self.activation4(self.bn4(x1))
        x1 = x1.reshape(x1.size(0), x1.size(1), x1.size(3), x1.size(4))
        e1 = self.SpectralSE(x1)
        x1 = torch.mul(e1, x1)

        # spatial
        x2 = self.conv5(x)
        x2 = self.activation5(self.bn5(x2))

        # Residual layer 2
        residual = x2
        residual = self.conv8(residual)
        x2 = self.conv6(x2)
        x2 = self.activation6(self.bn6(x2))
        x2 = self.conv7(x2)
        x2 = residual + x2

        x2 = self.activation7(self.bn7(x2))
        x2 = x2.reshape(x2.size(0), x2.size(1), x2.size(3), x2.size(4))
        e2 = self.SpatialSE(x2)
        x2 = torch.mul(e2,x2)

        # concat spatial and spectral information
        x = torch.cat((x1, x2), 1)

        x = self.avgpool(x)
        x = x.reshape((x.size(0), -1))

        # Fully connected layer
        # x = self.fc1(x)

        return x

#---------------------------------------------------------------------

from collections import OrderedDict
from einops import rearrange, repeat
def conv3x3x3(in_channel, out_channel):
    layer = nn.Sequential(
        nn.Conv3d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm3d(out_channel),
    )
    return layer

class residual_block(nn.Module):

    def __init__(self, in_channel, out_channel):
        super(residual_block, self).__init__()

        self.conv1 = conv3x3x3(in_channel, out_channel)
        self.conv2 = conv3x3x3(out_channel, out_channel)
        self.conv3 = conv3x3x3(out_channel, out_channel)

    def forward(self, x):  # (1,1,100,9,9)
        x1 = F.relu(self.conv1(x), inplace=True)
        x2 = F.relu(self.conv2(x1), inplace=True)
        x3 = self.conv3(x2)

        out = F.relu(x1 + x3, inplace=True)
        return out


class Residual(torch.nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(torch.nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = torch.nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(torch.nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, dim),
            torch.nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(torch.nn.Module):
    def __init__(self, dim, heads, dim_head, dropout):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = torch.nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = torch.nn.Sequential(
            torch.nn.Linear(inner_dim, dim),
            torch.nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        # x:[b,n,dim]
        b, n, _, h = *x.shape, self.heads

        # get qkv tuple:([b,n,head_num*head_dim],[...],[...])
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        # split q,k,v from [b,n,head_num*head_dim] -> [b,head_num,n,head_dim]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        # transpose(k) * q / sqrt(head_dim) -> [b,head_num,n,n]
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        # mask value: -inf
        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        # softmax normalization -> attention matrix
        attn = dots.softmax(dim=-1)
        # value * attention matrix -> output
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        # cat all output -> [b, n, head_num*head_dim]
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class ViT(torch.nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_head, dropout, num_channel):
        super().__init__()

        self.layers = torch.nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(torch.nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_head, dropout=dropout)))
            ]))

        self.skip_connection = torch.nn.ModuleList([])
        for _ in range(depth - 2):
            self.skip_connection.append(torch.nn.Conv2d(num_channel + 1, num_channel + 1, [1, 2], 1, 0))

    def forward(self, x, mask=None):
        last_output = []
        nl = 0
        for attn, ff in self.layers:
            last_output.append(x)
            if nl > 1:
                x = self.skip_connection[nl - 2](
                    torch.cat([x.unsqueeze(3), last_output[nl - 2].unsqueeze(3)], dim=3)).squeeze(3)
            x = attn(x, mask=mask)
            x = ff(x)
            nl += 1
        return x


class ImageEncoder(torch.nn.Module):#无分类器
    def __init__(self, patch_size, bands, num_classes, dim, depth, heads, mlp_dim, pool='cls',
                 embed_dim=512, dim_head=16, dropout=0., emb_dropout=0.):
        super().__init__()

        self.bands = bands
        self.patch_size = patch_size

        self.conv3d = residual_block(1, 8)
        self.x1 = self._get_layer_size()
        self.conv2d = nn.Sequential(
            nn.Conv2d(in_channels=self.x1.shape[1] * self.x1.shape[2], out_channels=bands, kernel_size=(3, 3),
                      padding=1),
            nn.ReLU(inplace=True))

        patch_dim = patch_size ** 2

        self.pos_embedding = torch.nn.Parameter(torch.randn(1, bands + 1, dim))
        self.patch_to_embedding = torch.nn.Linear(patch_dim, dim)
        self.cls_token = torch.nn.Parameter(torch.randn(1, 1, dim))

        self.dropout = torch.nn.Dropout(emb_dropout)
        self.vision_transformer = ViT(dim, depth, heads, dim_head, mlp_dim, dropout, bands)

        self.pool = pool
        self.to_latent = torch.nn.Identity()

        self.layer_norm = torch.nn.LayerNorm(dim)
        self.classification = torch.nn.Linear(dim, num_classes)
        self.fc = torch.nn.Linear(dim, embed_dim)


    def _get_layer_size(self):
        with torch.no_grad():
            x = torch.zeros((1, 1, self.bands,
                             self.patch_size, self.patch_size))
            s = self.conv3d(x)
        return s

    def forward(self, x, mask=None):
        x = x.unsqueeze(1)
        x = self.conv3d(x)
        x = rearrange(x, 'b c h w y -> b (c h) w y')
        x = self.conv2d(x)
        # patchs[batch, patch_num, patch_size*patch_size*c]  [batch,200,145*145]
        x = rearrange(x, 'b c h w -> b c (h w)')

        ## embedding every patch vector to embedding size: [batch, patch_num, embedding_size]
        x = self.patch_to_embedding(x)  # [b,n,dim]
        b, n, _ = x.shape

        # add position embedding
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)  # [b,1,dim]
        x = torch.cat((cls_tokens, x), dim=1)  # [b,n+1,dim]
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        # transformer: x[b,n + 1,dim] -> x[b,n + 1,dim]
        x = self.vision_transformer(x, mask)

        # classification: using cls_token output
        x = self.to_latent(x[:, 0])

        # MLP classification layer
        # return self.mlp_head(x)
        x = self.layer_norm(x)#-->x(256,64)
        # return self.classification(x), self.fc(x)
        return x


class TextEncode(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 context_length: 77,
                 vocab_size: 49408,
                 transformer_width: 512,
                 transformer_heads: 8,
                 transformer_layers: 3
                 ):
        super().__init__()
        #
        self.context_length = context_length
        #
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )
        #
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.linear_layer = nn.Linear(512, 152)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def encode_text(self, text):
        # text(56,77)
        x = self.token_embedding(text).type(dtype=torch.float32)  # [batch_size, n_ctx, d_model]
        # x（256,77,512）
        x = x + self.positional_embedding.type(dtype=torch.float32)
        x = x.permute(1, 0, 2)  # NLD -> LND
        # x（77,256,512）
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        # x（256,77,512）

        x = self.ln_final(x).type(dtype=torch.float32)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        # x(256,512)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)]
        x = x @ self.text_projection
        return x

    def forward(self, text, mask=None):
        text_features_1 = self.encode_text(text)
        text_features_1 = self.linear_layer(text_features_1)


        return text_features_1

class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.fc(x)
        #（256,512）->（256,512）
        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        #ret(77,256,512)
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


#---------------------------------------------------------------------

class RIPA_Module(nn.Module):

    def __init__(self, input_dim,beta_s=1.0, beta_t=1.0,
                 ):
        super().__init__()

        self.softmax = nn.Softmax(-1)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        pre_dim1 = input_dim
        pre_dim2 = input_dim
        self.beta_s = beta_s
        self.beta_t = beta_t
        self.scale = 0.1

        self.pre_project = nn.Sequential(  # 3 layers
            nn.Linear(input_dim, pre_dim1),
            nn.BatchNorm1d(pre_dim1),
            nn.ReLU(inplace=True),

            nn.Linear(pre_dim1, pre_dim2),
            nn.BatchNorm1d(pre_dim2),
            nn.ReLU(inplace=True),

            nn.Linear(pre_dim2, input_dim * 3)
        )

        self.post_project = nn.Sequential(  # only one layer
            nn.Linear(input_dim, input_dim)
        )

    def forward(self, Fv, Fvs_bank, Fvt_bank):

        out_fv = self.pre_project(Fv)  # (batch, 3 * C)
        out_fvs = self.pre_project(Fvs_bank)  # (N, 3 * C)
        out_fvt = self.pre_project(Fvt_bank)  # (N, 3 * C)

        q_fv, k_fv, v_fv = tuple(rearrange(out_fv, 'b (d k) -> k b d ', k=3))
        q_fvs, k_fvs, v_fvs = tuple(rearrange(out_fvs, 'b (d k) -> k b d ', k=3))
        q_fvt, k_fvt, v_fvt = tuple(rearrange(out_fvt, 'b (d k) -> k b d ', k=3))

        As = self.softmax(self.scale * q_fv @ k_fvs.permute(1, 0))  # (batch, N)
        At = self.softmax(self.scale * q_fv @ k_fvt.permute(1, 0))  # (batch, N)

        Fsta = Fv + self.post_project(As @ v_fvs) + self.post_project(At @ v_fvt) # (batch, C)

        Fsta = Fsta / Fsta.norm(dim=-1, keepdim=True)

        return Fsta

class SSSE_Houston(nn.Module):

    def __init__(self, input_channels, patch_size, n_classes):
        super(SSSE_Houston, self).__init__()
        self.kernel_dim = 1
        self.feature_dim = input_channels
        self.sz = patch_size
        # Convolution Layer 1 kernel_size = (1, 1, 7), stride = (1, 1, 2), output channels = 24
        self.conv1 = nn.Conv3d(1, 24, kernel_size=(7, 1, 1), stride=(2, 1, 1), bias=True)
        self.bn1 = nn.BatchNorm3d(24)
        self.activation1 = nn.ReLU()

        # Residual block 1
        self.conv2 = nn.Conv3d(24, 24, kernel_size=(7, 1, 1), stride=1, padding=(3, 0, 0), padding_mode='zeros',
                               bias=True)
        self.bn2 = nn.BatchNorm3d(24)
        self.activation2 = nn.ReLU()
        self.conv3 = nn.Conv3d(24, 24, kernel_size=(7, 1, 1), stride=1, padding=(3, 0, 0), padding_mode='zeros',
                               bias=True)
        self.bn3 = nn.BatchNorm3d(24)
        self.activation3 = nn.ReLU()
        # Finish

        # Convolution Layer 2 kernel_size = (1, 1, (self.feature_dim - 6) // 2), output channels = 128
        self.conv4 = nn.Conv3d(24, 128, kernel_size=(((self.feature_dim - 7) // 2 + 1), 1, 1), bias=True)
        self.bn4 = nn.BatchNorm3d(128)
        self.activation4 = nn.ReLU()
        if patch_size == 7:
            self.SpectralSE = SpectralSE(128, 128, self.sz)
        elif patch_size == 3:
            self.SpectralSE = SpectralSE_three(128, 128, self.sz)
        elif patch_size == 5:
            self.SpectralSE = SpectralSE_five(128, 128, self.sz)
        elif patch_size == 9:
            self.SpectralSE = SpectralSE_nine(128,128,self.sz)
        elif patch_size == 11:
            self.SpectralSE = SpectralSE_eleven(128, 128, self.sz)
        elif patch_size == 13:
            self.SpectralSE = SpectralSE_thirteen(128, 128, self.sz)
        elif patch_size == 1:
            self.SpectralSE = SpectralSE_one(128, 128, self.sz)
        else:
            raise Exception('this patch not define')
        # self.SpectralSE = SpectralSE_R(128, 128, self.sz)
        # self.SpectralSE = SpectralSE_S(128, 128, self.sz)

        # Convolution layer for spatial information
        self.conv5 = nn.Conv3d(1, 24, (self.feature_dim, 1, 1))
        self.bn5 = nn.BatchNorm3d(24)
        self.activation5 = nn.ReLU()

        # Residual block 2
        self.conv6 = nn.Conv3d(24, 24, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), padding_mode='zeros',
                               bias=True)
        self.bn6 = nn.BatchNorm3d(24)
        self.activation6 = nn.ReLU()
        self.conv7 = nn.Conv3d(24, 24, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), padding_mode='zeros',
                               bias=True)
        self.bn7 = nn.BatchNorm3d(24)
        self.activation7 = nn.ReLU()
        self.SpatialSE = SpatialSE(24, 1)
        self.conv8 = nn.Conv3d(24, 24, kernel_size=1)
        # Finish

        # Combination shape
        self.inter_size = 128 + 24

        # Residual block 3
        self.conv9 = nn.Conv3d(self.inter_size, self.inter_size, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1),
                               padding_mode='zeros', bias=True)
        self.bn9 = nn.BatchNorm3d(self.inter_size)
        self.activation9 = nn.ReLU()
        self.conv10 = nn.Conv3d(self.inter_size, self.inter_size, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1),
                                padding_mode='zeros', bias=True)
        self.bn10 = nn.BatchNorm3d(self.inter_size)
        self.activation10 = nn.ReLU()

        # Average pooling kernel_size = (5, 5, 1)
        self.avgpool = nn.AvgPool3d((1, self.sz, self.sz))
        # self.avgpool = nn.Conv2d(in_channels=self.inter_size,out_channels=self.inter_size,kernel_size=self.sz)

        # Fully connected Layer
        self.fc1 = nn.Linear(in_features=self.inter_size, out_features=n_classes)

        # parameters initialization
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, bounds=None):#x(32,48,7,7)
        # Convolution layer 1
        x = x.unsqueeze(1)#x(32,48,7,7)-->x(32,1,48,7,7)
        # spatial
        x1 = self.conv1(x)#-->x1(32,1,48,7,7)
        x1 = self.activation1(self.bn1(x1))
        # Residual layer 1
        residual = x1
        x1 = self.conv2(x1)
        x1 = self.activation2(self.bn2(x1))#-->x1(32,1,48,7,7)
        x1 = self.conv3(x1)
        x1 = residual + x1
        x1 = self.activation3(self.bn3(x1))#-->x1(32,1,48,7,7)

        # Convolution layer to combine rest
        x1 = self.conv4(x1)
        x1 = self.activation4(self.bn4(x1))#-->x1(32,128,1,7,7)
        x1 = x1.reshape(x1.size(0), x1.size(1), x1.size(3), x1.size(4))#-->x1(32,128,7,7)

        e1 = self.SpectralSE(x1)#-->e1(32,128,7,7)
        x1 = torch.mul(e1, x1)


        # spatial
        x2 = self.conv5(x)#-->x2(32,24,1,7,7)
        x2 = self.activation5(self.bn5(x2))

        # Residual layer 2
        residual = x2#-->residual(32,24,1,7,7)
        residual = self.conv8(residual)#-->residual(32,24,1,7,7)
        x2 = self.conv6(x2)#-->x2(32,24,1,7,7)
        x2 = self.activation6(self.bn6(x2))
        x2 = self.conv7(x2)
        x2 = residual + x2#-->x2(32,24,1,7,7)

        x2 = self.activation7(self.bn7(x2))#-->x2(32,24,1,7,7)
        x2 = x2.reshape(x2.size(0), x2.size(1), x2.size(3), x2.size(4))#-->x2(32,24,7,7)

        e2 = self.SpatialSE(x2)#-->e2(32,24,7,7)
        x2 = torch.mul(e2,x2)

        # concat spatial and spectral information
        x = torch.cat((x1, x2), 1)#-->x(32,128,7,7)

        x = self.avgpool(x)#-->x(32,24,1,1)
        x = x.reshape((x.size(0), -1))#-->x(32,152)

        # Fully connected layer
        # x = self.fc1(x)
        # print("x的形状",x.shape)
        return x


class SpatialSE(nn.Module):
    # 定义各个层的部分
    def __init__(self,in_channel,out_channel):
        super(SpatialSE, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        C = x.shape[1]
        x = self.conv(x)
        x = torch.sigmoid(x)
        out = x.repeat(1,C,1,1)
        return out


class SpectralSE_one(nn.Module):
    # 定义各个层的部分
    def __init__(self, in_channel, C, sz):
        super(SpectralSE_one, self).__init__()
        # 全局池化
        self.avgpool = nn.AvgPool2d((sz, sz))
        self.conv1 = nn.Conv2d(in_channel, C//4, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(C//4, C, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.avgpool(x)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        out = torch.sigmoid(x)
        return out

# patch size 7
class SpectralSE(nn.Module):
    # 定义各个层的部分
    def __init__(self, in_channel, C, sz):
        super(SpectralSE, self).__init__()
        # 全局池化
        self.avgpool = nn.AvgPool2d((sz, sz))
        self.conv1 = nn.Conv2d(in_channel, C//4, kernel_size=3, stride=1, padding=3)
        self.conv2 = nn.Conv2d(C//4, C, kernel_size=3, stride=1, padding=2)

    def forward(self, x):
        x = self.avgpool(x)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        out = torch.sigmoid(x)
        return out

# patch_size = 5
class SpectralSE_five(nn.Module):
    # 定义各个层的部分
    def __init__(self, in_channel, C, sz):
        super(SpectralSE_five, self).__init__()
        # 全局池化
        self.avgpool = nn.AvgPool2d((sz, sz))
        self.conv1 = nn.Conv2d(in_channel, C//4, kernel_size=3, stride=1, padding=2)
        self.conv2 = nn.Conv2d(C//4, C, kernel_size=3, stride=1, padding=2)

    def forward(self, x):
        x = self.avgpool(x)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        out = torch.sigmoid(x)
        return out

# patch_size = 9
class SpectralSE_nine(nn.Module):
    # 定义各个层的部分
    def __init__(self, in_channel, C, sz):
        super(SpectralSE_nine, self).__init__()
        # 全局池化
        self.avgpool = nn.AvgPool2d((sz, sz))
        self.conv1 = nn.Conv2d(in_channel, C//4, kernel_size=3, stride=1, padding=3)
        self.conv2 = nn.Conv2d(C//4, C, kernel_size=3, stride=1, padding=3)

    def forward(self, x):
        # x=(32,128,9,9)
        x = self.avgpool(x)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        out = torch.sigmoid(x)
        return out

# patch_size = 11
class SpectralSE_eleven(nn.Module):
    # 定义各个层的部分
    def __init__(self, in_channel, C, sz):
        super(SpectralSE_eleven, self).__init__()
        # 全局池化
        self.avgpool = nn.AvgPool2d((sz, sz))
        self.conv1 = nn.Conv2d(in_channel, C//4, kernel_size=3, stride=1, padding=3)
        self.conv2 = nn.Conv2d(C//4, C, kernel_size=3, stride=1, padding=4)

    def forward(self, x):
        x = self.avgpool(x)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        out = torch.sigmoid(x)
        return out

class SpectralSE_thirteen(nn.Module):
    # 定义各个层的部分
    def __init__(self, in_channel, C, sz):
        super(SpectralSE_thirteen, self).__init__()
        # 全局池化
        self.avgpool = nn.AvgPool2d((sz, sz))
        self.conv1 = nn.Conv2d(in_channel, C//4, kernel_size=3, stride=1, padding=4)
        self.conv2 = nn.Conv2d(C//4, C, kernel_size=3, stride=1, padding=4)

    def forward(self, x):
        x = self.avgpool(x)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        out = torch.sigmoid(x)
        return out


# patch_size = 3
class SpectralSE_three(nn.Module):
    # 定义各个层的部分
    def __init__(self, in_channel, C, sz):
        super(SpectralSE_three, self).__init__()
        # 全局池化
        self.avgpool = nn.AvgPool2d((sz, sz))
        self.conv1 = nn.Conv2d(in_channel, C//4, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(C//4, C, kernel_size=3, stride=1, padding=2)

    def forward(self, x):
        x = self.avgpool(x)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        out = torch.sigmoid(x)
        return out
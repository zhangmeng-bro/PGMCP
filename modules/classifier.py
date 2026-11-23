from typing import Tuple, Optional, List, Dict
import torch.nn as nn
import torch

__all__ = ['Classifier']


class Classifier(nn.Module):
    """A generic Classifier class for domain adaptation.

    Args:
        backbone (torch.nn.Module): Any backbone to extract 2-d features from data
        num_classes (int): Number of classes
        bottleneck (torch.nn.Module, optional): Any bottleneck layer. Use no bottleneck by default
        bottleneck_dim (int, optional): Feature dimension of the bottleneck layer. Default: -1
        head (torch.nn.Module, optional): Any classifier head. Use :class:`torch.nn.Linear` by default
        finetune (bool): Whether finetune the classifier or train from scratch. Default: True
        freeze (bool) : Freeze the backbone and only train the classifier

    .. note::
        Different classifiers are used in different domain adaptation algorithms to achieve better accuracy
        respectively, and we provide a suggested `Classifier` for different algorithms.
        Remember they are not the core of algorithms. You can implement your own `Classifier` and combine it with
        the domain adaptation algorithm in this algorithm library.

    .. note::
        The learning rate of this classifier is set 10 times to that of the feature extractor for better accuracy
        by default. If you have other optimization strategies, please over-ride :meth:`~Classifier.get_parameters`.

    Inputs:
        - x (tensor): input data fed to `backbone`

    Outputs:
        - predictions: classifier's predictions
        - features: features after `bottleneck` layer and before `head` layer

    Shape:
        - Inputs: (minibatch, *) where * means, any number of additional dimensions
        - predictions: (minibatch, `num_classes`)
        - features: (minibatch, `features_dim`)

    """

    # def __init__(self, backbone: nn.Module, num_classes: int, bottleneck: Optional[nn.Module] = None,
    #              bottleneck_dim: Optional[int] = -1, head: Optional[nn.Module] = None, finetune=True, pool_layer=None):

    # fineture = False.
    #用于构建分类器的模块。这个分类器由多个组件组成，包括主干网络（backbone）、瓶颈层（bottleneck）、池化层（pool_layer）和分类头（head）。
    def __init__(self, backbone: nn.Module, num_classes: int, bottleneck: Optional[nn.Module] = None,
                 bottleneck_dim: Optional[int] = -1, head: Optional[nn.Module] = None, finetune=True, pool_layer=None):

        super(Classifier, self).__init__()
        self.backbone = backbone#主干网络，通常是预训练的特征提取器（如 ResNet、VGG 等）。它负责提取输入数据的高级特征。
        self.num_classes = num_classes#分类器的目标类别数。
        if pool_layer is None:#池化层，用于将主干网络的特征图（feature map）压缩为固定大小的特征向量。如果为 None，则使用默认的全局平均池化层。
            self.pool_layer = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                nn.Flatten()
            )
        else:
            self.pool_layer = pool_layer
        if bottleneck is None:#瓶颈层（bottleneck layer），用于将主干网络的特征进一步降维或转换。如果为 None，则不使用瓶颈层。
            self.bottleneck = nn.Identity()
            self._features_dim = backbone.out_features
        else:
            self.bottleneck = bottleneck
            print("[INFORMATION] The bottleneck dim is ", bottleneck_dim)
            assert bottleneck_dim > 0
            self._features_dim = bottleneck_dim


        if head is None:
            self.head = nn.Linear(self._features_dim, num_classes)
        else:
            self.head = head
        self.finetune = finetune
          
    @property
    # 返回分类器中瓶颈层（bottleneck）的输出特征维度
    def features_dim(self) -> int:
        """The dimension of features before the final `head` layer"""
        return self._features_dim
    #实现了分类器的前向传播逻辑。这个方法接收输入张量 x，并返回分类器的预测结果。
    # 在训练模式下，它还会返回特征向量 f，这通常用于计算损失函数（如特征正则化或对比学习）
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """"""
        x= self.backbone(x)
        # x = self.backbone(x)
        f = self.pool_layer(x)
        f = self.bottleneck(f)
        predictions = self.head(f)
        if self.training:
            return predictions, f
        else:
            return predictions
    #用于生成分类器中各层的参数列表，并为每个参数组指定不同的学习率。
    def get_parameters(self, base_lr=1.0) -> List[Dict]:
        """A parameter list which decides optimization hyper-parameters,
            such as the relative learning rate of each layer
        """
        params = [
            {"params": self.backbone.parameters(), "lr": 0.1*base_lr if self.finetune else 1.0 * base_lr},
            {"params": self.bottleneck.parameters(), "lr": 1.0 * base_lr},
            {"params": self.head.parameters(), "lr": 1.0 * base_lr},
        ]

        return params

#继承自 Classifier 类，但没有添加任何新的属性或方法。这是一个非常简单的类定义，只是简单地继承了父类的所有功能
class ImageClassifier(Classifier):
    pass

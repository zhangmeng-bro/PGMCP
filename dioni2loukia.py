from __future__ import print_function
import argparse
import time
import torch
import torch.optim as optim
import wandb

import utils
from modules.mcc import MinimumClassConfusionLoss
from modules.teacher import EMATeacher
from utils import *
from basenet import *
import os
from HybridSN import SSSE_test,RIPA_Module
from loss_helper import *
from torch.utils.data import TensorDataset, DataLoader
from sklearn import metrics


# Training settings
parser = argparse.ArgumentParser(description='Visda Classification')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--num_epoch', type=int, default=50, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--optimizer', type=str, default='momentum', metavar='OP',
                    help='the name of optimizer')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=5, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--num_k', type=int, default=3, metavar='K',
                    help='how many steps to repeat the generator update')
parser.add_argument('--name', type=str, default='board', metavar='B',
                    help='board dir')

parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--patch_size', type=int, default=3,
                    help="Size of the spatial neighbourhood (optional, if "
                         "absent will be set by the model)")
parser.add_argument('--halfwidth', default=1, type=int)
parser.add_argument('--bank_num', default=5, type=int)
parser.add_argument('--training_sample', type=int, default=0.8,
                    help="The proportion of training source domain samples")
parser.add_argument('--training_tar_sample', type=int, default=0.3,
                    help="The proportion of training target domain samples(no labels)")
parser.add_argument('--large_num', type=int, default=3,
                    help="The data augmentation factor")
parser.add_argument('--num_trials', type=int, default=1,
                    help='the number of epoch')

parser.add_argument('--alpha', default=0.9, type=float)
parser.add_argument('--pseudo_label_weight', default="prob")

parser.add_argument('--norm-mean', type=float, nargs='+',
                    default=(0.485, 0.456, 0.406), help='normalization mean')
parser.add_argument('--norm-std', type=float, nargs='+',
                    default=(0.229, 0.224, 0.225), help='normalization std')

parser.add_argument('-i', '--iters-per-epoch', default=1000, type=int,
                    help='Number of iterations per epoch')


# project_name
parser.add_argument('--project_sub_name', type=str,
                    default="default", help="sub name of project")
parser.add_argument('--log_results', action='store_true',
                    help="To log results in wandb")
parser.add_argument('--log_name', type=str,
                    default="log", help="log name for wandb")

# Data augmentation parameters
group_da = parser.add_argument_group('Data augmentation')
group_da.add_argument('--flip_augmentation', action='store_true', default=True,
                      help="Random flips (if patch_size > 1)")
group_da.add_argument('--radiation_augmentation', action='store_true', default=True,
                      help="Random radiation noise (illumination)")
group_da.add_argument('--mixture_augmentation', action='store_true', default=False,
                      help="Random mixes between spectra")
# mcc loss
parser.add_argument('--temperature', default=8.0,
                        type=float, help='parameter temperature scaling')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
DEV = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
num_epoch = args.num_epoch
batch_size = args.batch_size
lr = args.lr
num_k = args.num_k
large_num = args.large_num
use_gpu = torch.cuda.is_available()

if __name__ == '__main__':
    acc_test_list1 = np.zeros([args.num_trials, args.num_trials])
    acc_test_list2 = np.zeros([args.num_trials, args.num_trials])

if args.log_results:
    os.environ['WANDB-API_KEY'] = ''
    os.environ['WANDB_MODE'] = ''
    run = wandb.init(
        project="MSDA",
        name=args.log_name)
    wandb.config.update(args)

# nDataSet = 10
num_classes = 12
N_BANDS = 176
HalfWidth = args.halfwidth
# BATCH_SIZE = 32
BATCH_SIZE = 32
patch_size = 2 * HalfWidth + 1

seeds = [4008,2004,3070,1067,4042,4048,3059,3069,1008,1049]#1
nDataSet = 10
acc = np.zeros([nDataSet, 1])
A = np.zeros([nDataSet, num_classes])
k = np.zeros([nDataSet, 1])

best_acc_all = 0.0
best_predict_all = 0
best_test_acc = 0
best_source_feature = []
best_source_label = []
best_target_feature = []
best_target_label = []
F1_correct = [0 for i in range(10)]
F2_correct = [0 for i in range(10)]
best_G, best_RandPerm, best_Row, best_Column, best_nTrain = None, None, None, None, None

def get_sample_data(Sample_data, Sample_label, HalfWidth, num_per_class):
    print('get_sample_data() run...')
    print('The original sample data shape:',Sample_data.shape)
    nBand = Sample_data.shape[2]

    data = np.pad(Sample_data, ((HalfWidth, HalfWidth), (HalfWidth, HalfWidth), (0, 0)), mode='constant')
    label = np.pad(Sample_label, HalfWidth, mode='constant')

    train = {}
    train_indices = []
    [Row, Column] = np.nonzero(label)
    m = int(np.max(label))
    print(f'num_class : {m}')

    val = {}
    val_indices = []

    for i in range(m):
        indices = [j for j, x in enumerate(Row.ravel().tolist()) if label[Row[j], Column[j]] == i + 1]
        np.random.shuffle(indices)
        num = len(indices)
        if num < num_per_class:
            train[i] = indices[:num_per_class]
            train[i] += indices[:num_per_class-num]
        else:
            train[i] = indices[:num_per_class]
        val[i] = indices[num_per_class:]

    for i in range(m):
        train_indices += train[i]
        val_indices += val[i]
    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)

    #val
    print('the number of val data:', len(val_indices))
    nVAL = len(val_indices)
    val_data = np.zeros([nVAL, nBand, 2 * HalfWidth + 1, 2 * HalfWidth + 1], dtype=np.float32)
    val_label = np.zeros([nVAL], dtype=np.int64)
    RandPerm = val_indices
    RandPerm = np.array(RandPerm)

    for i in range(nVAL):
        val_data[i, :, :, :] = np.transpose(data[Row[RandPerm[i]] - HalfWidth: Row[RandPerm[i]] + HalfWidth + 1, \
                                                  Column[RandPerm[i]] - HalfWidth: Column[RandPerm[i]] + HalfWidth + 1,
                                                  :],
                                                  (2, 0, 1))
        val_label[i] = label[Row[RandPerm[i]], Column[RandPerm[i]]].astype(np.int64)
    val_label = val_label - 1

    #train
    print('the number of processed data:', len(train_indices))
    nTrain = len(train_indices)
    index = np.zeros([nTrain], dtype=np.int64)
    processed_data = np.zeros([nTrain, nBand, 2 * HalfWidth + 1, 2 * HalfWidth + 1], dtype=np.float32)
    processed_label = np.zeros([nTrain], dtype=np.int64)
    RandPerm = train_indices
    RandPerm = np.array(RandPerm)

    for i in range(nTrain):
        index[i] = i
        processed_data[i, :, :, :] = np.transpose(data[Row[RandPerm[i]] - HalfWidth: Row[RandPerm[i]] + HalfWidth + 1, \
                                          Column[RandPerm[i]] - HalfWidth: Column[RandPerm[i]] + HalfWidth + 1, :],
                                          (2, 0, 1))
        processed_label[i] = label[Row[RandPerm[i]], Column[RandPerm[i]]].astype(np.int64)
    processed_label = processed_label - 1

    print('sample data shape', processed_data.shape)
    print('sample label shape', processed_label.shape)
    print('get_sample_data() end...')
    return processed_data, processed_label#, val_data, val_label
for iDataSet in range(nDataSet):
    data_path_s = './HyRANK/Dioni.mat'
    label_path_s = './HyRANK/Dioni_gt_out68.mat'
    data_path_t = './HyRANK/Loukia.mat'
    label_path_t = './HyRANK/Loukia_gt_out68.mat'

    data_s, label_s = utils.load_data_hyrank(data_path_s, label_path_s)
    data_t, label_t = utils.load_data_hyrank(data_path_t, label_path_t)
    print('#######################idataset######################## ', iDataSet)

    num_classes = label_s.max()
    N_BANDS = data_s.shape[-1]
    hyperparams = vars(args)
    hyperparams.update({'n_classes': num_classes, 'n_bands': N_BANDS,
                        'device': DEV, 'center_pixel': False, 'supervision': 'full'})
    hyperparams = dict((k, v) for k, v in hyperparams.items() if v is not None)

    utils.seed_everything(seeds[iDataSet], use_deterministic=True)

    trainX, trainY = get_sample_data(data_s, label_s, HalfWidth, 180)
    testID, testX, testY, Gr, RandPerm, Row, Column = utils.get_all_data(data_t, label_t, HalfWidth)
    train_dataset = TensorDataset(torch.tensor(trainX,dtype=torch.float32), torch.tensor(trainY,dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(testX,dtype=torch.float32), torch.tensor(testY,dtype=torch.long))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    train_tar_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)


    len_src_loader = len(train_loader)
    len_tar_train_loader = len(train_tar_loader)
    len_src_dataset = len(train_loader.dataset)
    len_tar_train_dataset = len(train_tar_loader.dataset)
    len_tar_dataset = len(test_loader.dataset)
    len_tar_loader = len(test_loader)

    G = SSSE_test(input_channels=N_BANDS,
                  patch_size=2 * HalfWidth + 1,
                  n_classes=num_classes).to(DEV)
    RIPA = RIPA_Module(input_dim=152, beta_s=0.1, beta_t=0.1).to(DEV)
    F1 = ResClassifiervit1(num_classes, training=True)
    F2 = ResClassifiervit1(num_classes, training=True)

    F1.apply(weights_init)
    F2.apply(weights_init)

    # EMA teacher
    F1_teacher = EMATeacher(G, F1, alpha=args.alpha, pseudo_label_weight=args.pseudo_label_weight).to(DEV)


    # mcc
    mcc_loss = MinimumClassConfusionLoss(temperature=args.temperature)

    bank_num = args.bank_num
    source_key_dict = {i: i for i in range(num_classes * bank_num)}
    target_key_dict = {i: i for i in range(num_classes * bank_num)}
    source_max_probs_list = [0.0 for i in range(num_classes * bank_num)]
    target_max_probs_list = [0.0 for i in range(num_classes * bank_num)]
    source_feat_bank = torch.zeros((num_classes * bank_num, 152))
    target_feat_bank = torch.zeros((num_classes * bank_num, 152))
    source_feat_bank = torch.zeros((num_classes * bank_num, 152))
    target_feat_bank = torch.zeros((num_classes * bank_num, 152))
    source_feat_bank = nn.Parameter(source_feat_bank, requires_grad=False)
    target_feat_bank = nn.Parameter(target_feat_bank, requires_grad=False)
    source_bias = nn.Parameter(torch.zeros(num_classes, 152).to(DEV))
    target_bias = nn.Parameter(torch.zeros(num_classes, 152).to(DEV))

    lr = args.lr
    if args.cuda:
        G.to(DEV)
        RIPA.to(DEV)
        F1.to(DEV)
        F2.to(DEV)
    if args.optimizer == 'momentum':
        optimizer_g = optim.SGD(list(G.parameters()), lr=args.lr, weight_decay=0.0005)
        optimizer_f = optim.SGD(list(F1.parameters()) + list(F2.parameters()), momentum=0.9, lr=args.lr,
                                weight_decay=0.0005)
    elif args.optimizer == 'adam':
        optimizer_g = optim.Adam(G.features.parameters(), lr=args.lr, weight_decay=0.0005)
        optimizer_f = optim.Adam(list(F1.parameters()) + list(F2.parameters()), lr=args.lr, weight_decay=0.0005)
    else:
        optimizer_g = optim.Adadelta(G.features.parameters(), lr=args.lr, weight_decay=0.0005)
        optimizer_f = optim.Adadelta(list(F1.parameters()) + list(F2.parameters()), lr=args.lr, weight_decay=0.0005)

    def train(ep, train_loader, train_tar_loader):
        iter_source, iter_target = iter(train_loader), iter(train_tar_loader)
        criterion = nn.CrossEntropyLoss().to(DEV)
        G.train()
        F1.train()
        F2.train()
        num_iter = len_src_loader
        for batch_idx in range(1, num_iter):
            if batch_idx % len(train_tar_loader) == 0:
                iter_target = iter(train_tar_loader)
            data_source, label_source = next(iter_source)
            data_target, _ = next(iter_target)
            # label_source = label_source - 1
            if args.cuda:
                data1, target1 = data_source.to(DEV), label_source.to(DEV)
                data2 = data_target.to(DEV)
            # when pretraining network source only
            eta = 1.0
            data = Variable(torch.cat((data1, data2), 0))
            target1 = Variable(target1)
            # f1
            F1_teacher.update_weights(G, F1, ep * num_iter + batch_idx)
            F1_pseudo_label_t, F1_pseudo_prob_t = F1_teacher(data2)

            # Step A train all networks to minimize loss on source
            optimizer_g.zero_grad()
            optimizer_f.zero_grad()
            all_image_features= G(data)
            src_image_features = G(data1)
            tar_image_features = G(data2)
            output1 = F1(all_image_features)
            output2 = F2(all_image_features)

            image_prob_src1 = output1[0][:batch_size, :]
            image_prob_src2 = output2[0][:batch_size, :]
            image_prob_tar1 = output1[0][batch_size:, :]
            image_prob_tar2 = output2[0][batch_size:, :]
            output_t1 = F.softmax(image_prob_tar1, dim=1)
            output_t2 = F.softmax(image_prob_tar2, dim=1)
            pseudo_src_label = torch.softmax(image_prob_src1, dim=-1)
            max_probs_src, label_p_src = torch.max(pseudo_src_label, dim=-1)
            for i, l in enumerate(target1):
                if l == label_p_src[i]:
                    index = l.item() * bank_num
                    l_list = source_max_probs_list[index: index + bank_num]
                    if max_probs_src[i] > min(l_list):
                        min_index = l_list.index(min(l_list))
                        with torch.no_grad():
                            source_max_probs_list[index + min_index] = max_probs_src[i].detach().clone()
                            source_feat_bank[index + min_index] = src_image_features[i].detach().clone()
                            source_key_dict[index + min_index] = label_p_src[i].detach().clone()
                    elif ep > 1:
                        min_index = random.randint(0, len(l_list) - 1)
                        with torch.no_grad():
                            source_max_probs_list[index + min_index] = max_probs_src[i].detach().clone()
                            source_feat_bank[index + min_index] = src_image_features[i].detach().clone()
                            source_key_dict[index + min_index] = label_p_src[i].detach().clone()
            pseudo_tar_label = torch.softmax(image_prob_tar1, dim=-1)
            max_probs_tar, label_p_tar = torch.max(pseudo_tar_label, dim=-1)
            for i, l in enumerate(label_p_tar):
                index = l.item() * bank_num
                l_list = target_max_probs_list[index: index + bank_num]
                if max_probs_tar[i] > min(l_list):
                    min_index = l_list.index(min(l_list))
                    with torch.no_grad():
                        target_max_probs_list[index + min_index] = max_probs_tar[i].detach().clone()
                        target_feat_bank[index + min_index] = tar_image_features[i].detach().clone()
                        target_key_dict[index + min_index] = label_p_tar[i].detach().clone()
                elif ep > 1:
                    min_index = random.randint(0, len(l_list) - 1)
                    with torch.no_grad():
                        target_max_probs_list[index + min_index] = max_probs_tar[i].detach().clone()
                        target_feat_bank[index + min_index] = tar_image_features[i].detach().clone()
                        target_key_dict[index + min_index] = label_p_tar[i].detach().clone()
            source_bank = torch.mean(source_feat_bank.reshape(num_classes, bank_num, 152), dim=1).to(DEV) + source_bias
            target_bank = torch.mean(target_feat_bank.reshape(num_classes, bank_num, 152), dim=1).to(DEV) + target_bias

            loss1 = criterion(image_prob_src1, target1.long())
            loss2 = criterion(image_prob_src2, target1.long())
            IFT_loss = 0
            IFT_tar_image = RIPA(tar_image_features,source_bank, target_bank)
            IFT_scr_image = RIPA(src_image_features, source_bank, target_bank)
            y_t_A_f1_IFT, _ = F1(IFT_tar_image)
            y_s_f1_IFT, _ = F1(IFT_scr_image)
            loss3 = criterion(y_s_f1_IFT, target1.long())
            if F1_teacher.pseudo_label_weight is not None:
                ce = F.cross_entropy(y_t_A_f1_IFT, F1_pseudo_label_t, reduction='none').float()
                IFT_loss += torch.mean(F1_pseudo_prob_t * ce)
            else:
                IFT_loss += F.cross_entropy(y_t_A_f1_IFT, F1_pseudo_label_t)

            mcc_loss_value = (mcc_loss(output_t1) + mcc_loss(output_t2))
            if ep<=25:
                all_loss = loss1 + loss2 + loss3 + mcc_loss_value
            else:
                all_loss = loss1 + loss2 + loss3 + mcc_loss_value + IFT_loss

            if args.log_results:
                  wandb.log({f'{seeds[iDataSet]}_A_loss': all_loss,
                           f'{seeds[iDataSet]}_A_IFT_loss': IFT_loss})
            all_loss.backward()
            optimizer_g.step()
            optimizer_f.step()

            # Step B train classifier to maximize discrepancy
            optimizer_g.zero_grad()
            optimizer_f.zero_grad()

            all_image_features= G(data)
            output1 = F1(all_image_features)
            output2 = F2(all_image_features)
            image_prob_src1 = output1[0][:batch_size, :]
            image_prob_src2 = output2[0][:batch_size, :]
            image_prob_tar1 = output1[0][batch_size:, :]
            image_prob_tar2 = output2[0][batch_size:, :]
            output_t1 = F.softmax(image_prob_tar1, dim=1)
            output_t2 = F.softmax(image_prob_tar2, dim=1)
            loss1 = criterion(image_prob_src1, target1.long())
            loss2 = criterion(image_prob_src2, target1.long())

            loss_dis = torch.mean(torch.abs(output_t1 - output_t2))
            # mcc_loss_value = 0.0
            mcc_loss_value = (mcc_loss(output_t1) + mcc_loss(output_t2))

            F_loss = loss1 + loss2 - eta * loss_dis  + mcc_loss_value
            if args.log_results:
                wandb.log({f'{seeds[iDataSet]}_B_loss': F_loss})
            F_loss.backward()
            optimizer_f.step()
            # Step C train genrator to minimize discrepancy
            for i in range(num_k):
                optimizer_g.zero_grad()
                all_image_features = G(data)

                output1 = F1(all_image_features)
                output2 = F2(all_image_features)
                image_prob_src1 = output1[0][:batch_size, :]
                image_prob_src2 = output2[0][:batch_size, :]
                image_prob_tar1 = output1[0][batch_size:, :]
                image_prob_tar2 = output2[0][batch_size:, :]

                loss1 = criterion(image_prob_src1, target1.long())
                loss2 = criterion(image_prob_src2, target1.long())
                output_t1 = F.softmax(image_prob_tar1, dim=1)
                output_t2 = F.softmax(image_prob_tar2, dim=1)

                mcc_loss_value = (mcc_loss(output_t1) + mcc_loss(output_t2))
                loss_dis = torch.mean(torch.abs(output_t1 - output_t2)) + mcc_loss_value

                if args.log_results:
                    wandb.log({f'{seeds[iDataSet]}_C_loss': loss_dis})
                loss_dis.backward()
                optimizer_g.step()
            if batch_idx % args.log_interval == 0:
                print(
                    'Train Ep: {} [{}/{} ({:.0f}%)]\tLoss1: {:.6f}\tLoss2: {:.6f}\tLoss3: {:.6f}\t Dis: {:.6f}  MCC: {:.6f}  IFT_loss: {:.6f} '.format(
                        ep, batch_idx * len(data), args.batch_size * len(train_loader),
                            100. * batch_idx / len(train_loader), loss1.item(), loss2.item(),loss3.item(), loss_dis.item(),
                        mcc_loss_value, IFT_loss.item()))

            if batch_idx == 1 and ep > 1:
                G.train()
                F1.train()
                F2.train()


    def test(test_loader):
        G.eval()
        F1.eval()
        F2.eval()
        test_loss = 0
        correct = 0
        correct2 = 0
        size = 0
        predict = np.array([], dtype=np.int64)
        labels = np.array([], dtype=np.int64)
        pred1_list, pred2_list, label_list, outputdata = [], [], [], []
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(DEV), target.to(DEV)
                target2 = target
                data1, target1 = Variable(data), Variable(target2)
                output = G(data1)
                outputdata.append(output.cpu().numpy())
                output1 = F1(output)
                output2 = F2(output)
                test_loss += F.nll_loss(output1, target1.long()).item()

                pred1 = output1.data.max(1)[1]  # get the index of the max log-probability
                correct += pred1.eq(target1.data).cpu().sum()
                pred2 = output2.data.max(1)[1]  # get the index of the max log-probability
                correct2 += pred2.eq(target1.data).cpu().sum()
                k = target1.data.size()[0]
                pred1_list.append(pred1.cpu().numpy())
                pred2_list.append(pred2.cpu().numpy())
                predict = np.append(predict, pred1.cpu().numpy())
                labels = np.append(labels, target.cpu().numpy())
                label_list.append(target2.cpu().numpy())
                size += k
                acc1 = 100. * float(correct) / float(size)
                acc2 = 100. * float(correct2) / float(size)

            test_loss = test_loss
            test_loss /= len(test_loader)  # loss function already averages over batch size
            F1_correct[iDataSet]=100. * correct / len_tar_dataset
            F2_correct[iDataSet]=100. * correct2 / len_tar_dataset
            print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%) ({:.2f}%)\n'.format(
                test_loss, correct, len_tar_dataset,
                100. * correct / len_tar_dataset, 100. * correct2 / len_tar_dataset))

            # if 100. * correct / size > 67 or 100. * correct2 / size > 67:
            value = max(100. * correct / len_tar_dataset, 100. * correct2 / len_tar_dataset)
            if args.log_results:
                wandb.log({f'{seeds[iDataSet]}_result_1': 100. * correct / len_tar_dataset,
                           f'{seeds[iDataSet]}_result_2': 100. * correct2 / len_tar_dataset})
        return value, pred1_list, pred2_list, label_list, acc1, acc2, outputdata, label_list, predict, labels





    train_start = time.time()
    for ep in range(1, num_epoch + 1):
        train(ep, train_loader, train_tar_loader)
        # correct = val(val_loader)
        # 5 epoch test
        if ep % num_epoch == 0:
            test_start = time.time()
            value, pred1_list, pred2_list, label_list, acc1, acc2, outputdata_target, target_label, predict, labels = test(
                test_loader)
            test_end = time.time()
            acc[iDataSet] = acc1
            C = metrics.confusion_matrix(labels, predict)
            A[iDataSet, :] = np.diag(C) / np.sum(C, 1, dtype=np.float64)
            k[iDataSet] = metrics.cohen_kappa_score(labels, predict)
            if acc1 >= best_test_acc:
                best_test_acc = acc1
                best_predict_all = predict
                best_G, best_RandPerm, best_Row, best_Column = Gr, RandPerm, Row, Column

    train_end = time.time()


AA = np.mean(A, 1)
AAMean = np.mean(AA, 0)
AAStd = np.std(AA)
AMean = np.mean(A, 0)
AStd = np.std(A, 0)
OAMean = np.mean(acc)
OAStd = np.std(acc)
kMean = np.mean(k)
kStd = np.std(k)
print("train time per DataSet(s): " + "{:.5f}".format(train_end - train_start))
print("test time per DataSet(s): " + "{:.5f}".format(test_end - test_start))
print("average OA: " + "{:.2f}".format(OAMean) + " +- " + "{:.2f}".format(OAStd))
print("average AA: " + "{:.2f}".format(100 * AAMean) + " +- " + "{:.2f}".format(100 * AAStd))
print("average kappa: " + "{:.4f}".format(100 * kMean) + " +- " + "{:.4f}".format(100 * kStd))
print("accuracy for each class: ")
for i in range(num_classes):
    print("Class " + str(i) + ": " + "{:.2f}".format(100 * AMean[i]) + " +- " + "{:.2f}".format(100 * AStd[i]))
print("F1_correct:",  ",".join(map(str, F1_correct)))
print("F2_correct:",  ",".join(map(str, F2_correct)))




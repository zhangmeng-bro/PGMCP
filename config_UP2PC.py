nDataSet = 1

BATCH_SIZE = 128
epochs = 300
lr = 0.05
lrf = 0.01
lrpk = 0.05
CLASS_NUM = 7
nBand = 102
HalfWidth = 3
train_num = 40

momentum = 0.9
patch_size = 2 * HalfWidth + 1
no_cuda = False
cuda_id = '0'
l2_decay = 5e-4
seeds = [1104, 1105, 1117, 1128, 1131, 1135, 1138, 1149, 1165, 1166, 1179]
train_end = 0
test_end = 0

pk_prior_weight = 10.
pk_type = 'ub'  ####ub br
pk_uconf = 1.0
pk_knn = 1
distance = 'cosine'
ema = 0.6
turnnum = 3
interval_iter = 40
a = 0.3


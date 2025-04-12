
import random
from torch.utils.data import Dataset
import tensorly as tl
from tensorly.random import check_random_state
from MDD_data_set import *
import torch.nn as nn
import torch.optim as optim
import util
from tensorly.tucker_tensor import tucker_to_tensor
from scipy.io import savemat
# set seed and device
torch.manual_seed(112)  #122
np.random.seed(112)
random.seed(112)
if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.cuda.manual_seed_all(12)
else:
    device = torch.device("cpu")


class Transformer1(nn.Module):
    def __init__(self):
        super(Transformer1, self).__init__()
        self.dense_1 =nn.Linear(192, 16)
        self.dense_2 =nn.Linear(32, 2)
        self.fc_p = nn.Linear(3, 16)
        self.relu =nn.ReLU()
        self.l = nn.Linear(116,32)

    def forward(self, p, enc_inputs, g, f, l):
        x = self.dense_1(enc_inputs)
        p = self.fc_p(p)
        a1,a2,a3 = l.shape
        #l = l.view(a1, a2 * a3)
        l = l.sum(dim=1)
        l = self.l(l)
        #x = self.relu(x)
        #p = self.relu(p)
        f_o = g*(torch.cat((x,p),dim=1)) + (1 - g) * l
        #f_o = torch.cat((p, x), dim=1)
        x = self.dense_2(f_o)
        return x

class Transformer2(nn.Module):
    def __init__(self):
        super(Transformer2, self).__init__()
        self.dense_1 =nn.Linear(192, 16)
        self.dense_2 =nn.Linear(32, 2)
        self.fc_p = nn.Linear(3, 16)
        self.relu =nn.ReLU()
        self.l = nn.Linear(116, 32)

    def forward(self, p, enc_inputs, g, f,l):
        x = self.dense_1(enc_inputs)
        p = self.fc_p(p)
        a1, a2, a3 = l.shape
        #l = l.view(a1, a2 * a3)
        l = l.sum(dim=1)
        l = self.l(l)
       # x = self.relu(x)
       # p = self.relu(p)
        f_o = g*(torch.cat((x,p),dim=1)) + (1 - g) * l
        x = self.dense_2(f_o)
        return x

class Transformer3(nn.Module):
    def __init__(self):
        super(Transformer3, self).__init__()
        self.dense_1 =nn.Linear(192, 16)
        self.dense_2 =nn.Linear(32, 2)
        self.fc_p = nn.Linear(3, 16)
        self.relu =nn.ReLU()
        self.l = nn.Linear(116, 32)

    def forward(self, p, enc_inputs, g, f,l):
        x =self.dense_1(enc_inputs)
        p = self.fc_p(p)
        a1, a2, a3 = l.shape
        #l = l.view(a1, a2 * a3)
        l = l.sum(dim=1)
        l = self.l(l)
       # x = self.relu(x)
        #p = self.relu(p)
        f_o = g*(torch.cat((x,p),dim=1)) + (1 - g) * l
        x =self.dense_2(f_o)
        return x
save_root_path = r'D:\2024博士工作\FL\code\FLcode\result'

dataset1 = DatasetHCPRest1(k_fold=5)
dataloader1 = torch.utils.data.DataLoader(dataset1, batch_size=600, shuffle=False, num_workers=0, pin_memory=True)
dataset2 = DatasetHCPRest2(k_fold=5)
dataloader2 = torch.utils.data.DataLoader(dataset2, batch_size=600, shuffle=False, num_workers=0, pin_memory=True)
dataset3 = DatasetHCPRest3(k_fold=5)
dataloader3 = torch.utils.data.DataLoader(dataset3, batch_size=600, shuffle=False, num_workers=0, pin_memory=True)

random_state = 123 #123
rng = check_random_state(random_state)
rank = [0, 6, 8, 8]
modes = [0, 1, 2, 3]
factors1 = tl.tensor(rng.random_sample((3, rank[1])))
factors2 = tl.tensor(rng.random_sample((116, rank[2])))
factors3 = tl.tensor(rng.random_sample((116, rank[3])))

Model11 = STAR1(40, 116)
parameter_init1 = Model11.state_dict()
parameter_init_fix1 = Model11.state_dict()

Model22 = STAR2(40, 116)
parameter_init2 = Model22.state_dict()
parameter_init_fix2 = Model22.state_dict()

Model33 = STAR3(40, 116)
parameter_init3 = Model33.state_dict()
parameter_init_fix3 = Model33.state_dict()

window_size = 40
stride = 2

def sliding_window(input, window_size, stride):
    B, D, L = input.shape
    windows = [input[:, :, i:i + window_size] for i in range(0, L - window_size + 1, stride)]
    return torch.stack(windows, dim=1)  # shape (B, num_windows, D, window_size)

n_epoch = 15
k_fold = 5
test_acc11=[]
test_acc22=[]
test_acc33=[]
logger1 = util.logger.LoggerMDGL(5, 2)
logger2 = util.logger.LoggerMDGL(5, 2)
logger3 = util.logger.LoggerMDGL(5, 2)
for k in range(k_fold):

    model1 = Transformer1()
    learning_rate = 0.01
    criterion1 = torch.nn.CrossEntropyLoss()
    optimizer1 = optim.Adam(model1.parameters(), lr=learning_rate)

    model2 = Transformer2()
    criterion2 = torch.nn.CrossEntropyLoss()
    optimizer2 = optim.Adam(model2.parameters(), lr=learning_rate)

    model3 = Transformer3()
    criterion3 = torch.nn.CrossEntropyLoss()
    optimizer3 = optim.Adam(model3.parameters(), lr=learning_rate)
    g = 0.9

    for epoch in range(n_epoch):
        para_list = []
        dataset1.set_fold(k, train=True)
        dataset2.set_fold(k, train=True)
        dataset3.set_fold(k, train=True)
        train_acc1 = 0.0
        train_loss1 = 0.0
        test_acc1 = 0.0
        test_loss1 = 0.0
        train_acc2 = 0.0
        train_loss2 = 0.0
        test_acc2 = 0.0
        test_loss2 = 0.0
        train_acc3 = 0.0
        train_loss3 = 0.0
        test_acc3 = 0.0
        test_loss3 = 0.0
        model1.train()
        model2.train()
        model3.train()
        Model11.train(), Model22.train(), Model33.train()
        for site1 in range(1, 4):

            for _, x in enumerate(dataloader1):
                data1 = x['pc1'].float()
                timeseries1 = x['timeseries1'].float()
                people1 = x['people1'].float()
                label1 = x['label1']
                if epoch == 0 and site1 == 1:
                    # 初始化 factors
                    factors01 = np.identity(data1.size(0))
                    factors11 = [factors01] + [factors1] + [factors2] + [factors3]
                else:
                    factors01 = np.identity(data1.size(0))
                    factors11[0] = factors01
                US1 = getUS(tl.tensor(data1), factors11, site1)

            for _, x in enumerate(dataloader2):
                data2 = x['pc2'].float()
                timeseries2 = x['timeseries2'].float()
                people2 = x['people2'].float()
                label2 = x['label2']
                if epoch == 0 and site1 == 1:
                    # 初始化 factors
                    factors02 = np.identity(data2.size(0))
                    factors22 = [factors02] + [factors1] + [factors2] + [factors3]
                else:
                    factors02 = np.identity(data2.size(0))
                    factors22[0] = factors02
                US2 = getUS(tl.tensor(data2), factors22, site1)

            for _, x in enumerate(dataloader3):
                data3 = x['pc3'].float()
                timeseries3 = x['timeseries3'].float()
                people3 = x['people3'].float()
                label3 = x['label3']
                if epoch == 0 and site1 == 1:
                    # 初始化 factors
                    factors03 = np.identity(data3.size(0))
                    factors33 = [factors03] + [factors1] + [factors2] + [factors3]
                else:
                    factors03 = np.identity(data3.size(0))
                    factors33[0] = factors03
                US3 = getUS(tl.tensor(data3), factors33, site1)

            MH1 = np.hstack((US1, US2, US3))
            eigenvecs, sh, _ = np.linalg.svd(MH1)
            temp = eigenvecs[:, 0: rank[site1]]
            factors11[site1] = temp
            factors22[site1] = temp
            factors33[site1] = temp

            if site1 == 3:
                core1 = multi_mode_dot(tl.tensor(data1), factors11, modes=modes, transpose=True)
                core2 = multi_mode_dot(tl.tensor(data2), factors22, modes=modes, transpose=True)
                core3 = multi_mode_dot(tl.tensor(data3), factors33, modes=modes, transpose=True)

                rec1 = tucker_to_tensor(core1, factors11)
                rec2 = tucker_to_tensor(core2, factors22)
                rec3 = tucker_to_tensor(core3, factors33)


        print(epoch)

        nyu = 'site20.mat'
        un = 'site21.mat'
        lunven = 'site25.mat'
        if epoch == n_epoch - 1:
            savemat(nyu, {'a1': rec1, 'site20lable': label1})
            savemat(un, {'a2': rec2, 'site21lable': label2})
            savemat(lunven, {'a3': rec3, 'site25lable': label3})
        # if epoch == n_epoch - 1:
        #     torch.save(Model11.state_dict(),
        #                save_root_path + '/' + 'site' + str(epoch) + '_fold' + str(k) + '_model.pth')











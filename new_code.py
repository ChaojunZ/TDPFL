
import random
from torch.utils.data import Dataset
import tensorly as tl
from tensorly.random import check_random_state
from tqdm import tqdm
from N_data_set import *
import torch.nn as nn
import torch.optim as optim
from ot import emd

# set seed and device
torch.manual_seed(122)  #122
np.random.seed(122)
random.seed(122)
if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.cuda.manual_seed_all(12)
else:
    device = torch.device("cpu")


class Transformer1(nn.Module):
    def __init__(self):
        super(Transformer1, self).__init__()
        self.dense_1 =nn.Linear(147, 16)
        self.dense_2 =nn.Linear(32, 2)
        self.fc_p = nn.Linear(3, 16)
        self.relu =nn.ReLU()
        self.l = nn.Linear(116,32)

    def forward(self, p, enc_inputs, g, f, l):
        x = self.dense_1(enc_inputs)
        p = self.fc_p(p)
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
        self.dense_1 =nn.Linear(147, 16)
        self.dense_2 =nn.Linear(32, 2)
        self.fc_p = nn.Linear(3, 16)
        self.relu =nn.ReLU()
        self.l = nn.Linear(116, 32)

    def forward(self, p, enc_inputs, g, f,l):
        x = self.dense_1(enc_inputs)
        p = self.fc_p(p)
        l = self.l(l)
       # x = self.relu(x)
       # p = self.relu(p)
        f_o = g*(torch.cat((x,p),dim=1)) + (1 - g) * l
        x = self.dense_2(f_o)
        return x

class Transformer3(nn.Module):
    def __init__(self):
        super(Transformer3, self).__init__()
        self.dense_1 =nn.Linear(147, 16)
        self.dense_2 =nn.Linear(32, 2)
        self.fc_p = nn.Linear(3, 16)
        self.relu =nn.ReLU()
        self.l = nn.Linear(116, 32)

    def forward(self, p, enc_inputs, g, f,l):
        x =self.dense_1(enc_inputs)
        p = self.fc_p(p)
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
rank = [0, 5, 7, 7]
modes = [0, 1, 2, 3]
factors1 = tl.tensor(rng.random_sample((3, rank[1])))
factors2 = tl.tensor(rng.random_sample((116, rank[2])))
factors3 = tl.tensor(rng.random_sample((116, rank[3])))

Model11 = STAR1(175, 116)
parameter_init1 = Model11.state_dict()
parameter_init_fix1 = Model11.state_dict()

Model22 = STAR2(175, 116)
parameter_init2 = Model22.state_dict()
parameter_init_fix2 = Model22.state_dict()

Model33 = STAR3(175, 116)
parameter_init3 = Model33.state_dict()
parameter_init_fix3 = Model33.state_dict()



n_epoch = 15
k_fold = 5
test_acc11=[]
test_acc22=[]
test_acc33=[]
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
    g = 0.8

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
                label1 = x['label1'].float()
                B, D, L = timeseries1.size()
                d_series1 = L
                d_core1 = D
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
                label2 = x['label2'].float()
                B, D, L = timeseries2.size()
                d_series2 = L
                d_core2 = D
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
                label3 = x['label3'].float()
                B, D, L = timeseries3.size()
                d_series3 = L
                d_core3 = D
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

                Model11 = STAR1(d_series1, d_core1)
                Model22 = STAR2(d_series2, d_core2)
                Model33 = STAR3(d_series3, d_core3)
                Model11.load_state_dict(parameter_init1)  # read server-side parameters
                Model22.load_state_dict(parameter_init2)  # read server-side parameters
                Model33.load_state_dict(parameter_init3)  # read server-side parameters

                d1 = torch.from_numpy(core1.reshape(core1.shape[0], -1)).float()
                optimizer1.zero_grad()
                output1, score1 = Model11(timeseries1)
                output = model1(people1, d1, g, output1, score1)
                batch_loss1 = criterion1(output, label1.long())
                _, train_pred = torch.max(output, 1)
                batch_loss1.backward()
                optimizer1.step()
                train_acc1 += (train_pred.cpu() == label1.cpu()).sum().item()
                train_loss1 += batch_loss1.item()

                d2 = torch.from_numpy(core2.reshape(core2.shape[0], -1)).float()
                optimizer2.zero_grad()
                output2, score2 = Model22(timeseries2)
                output = model2(people2, d2, g, output2, score2)
                batch_loss2 = criterion2(output, label2.long())
                _, train_pred = torch.max(output, 1)
                batch_loss2.backward()
                optimizer2.step()
                train_acc2 += (train_pred.cpu() == label2.cpu()).sum().item()
                train_loss2 += batch_loss2.item()

                d3 = torch.from_numpy(core3.reshape(core3.shape[0], -1)).float()
                optimizer3.zero_grad()
                output3, score3 = Model33(timeseries3)
                output = model3(people3, d3, g, output3, score3)
                batch_loss3 = criterion3(output, label3.long())
                _, train_pred = torch.max(output, 1)
                batch_loss3.backward()
                optimizer3.step()
                train_acc3 += (train_pred.cpu() == label3.cpu()).sum().item()
                train_loss3 += batch_loss3.item()

        # if epoch == n_epoch - 1:
        #     torch.save(Model11.state_dict(),
        #                save_root_path + '/' + 'site' + str(epoch) + '_fold' + str(k) + '_model.pth')

        para_list.append(Model11.state_dict())
        para_list.append(Model22.state_dict())
        para_list.append(Model33.state_dict())

        parameter_init1 = W_A(para_list, [184, 66, 61])
        parameter_init2 = W_A(para_list, [184, 66, 61])
        parameter_init3 = W_A(para_list, [184, 66, 61])



        model1.eval()
        model2.eval()
        model3.eval()
        Model11.eval(), Model22.eval(), Model33.eval()
        dataset1.set_fold(k, train=False)
        dataset2.set_fold(k, train=False)
        dataset3.set_fold(k, train=False)
        with torch.no_grad():
            for _, x in enumerate(dataloader1):
                tdata1 = x['pc1'].float()
                people1 = x['people1'].float()
                timeseries1 = x['timeseries1'].float()
                tlabel1 = x['label1'].float()
                tfactors01 = np.identity(tdata1.size(0))
                tfactors11 = factors11
                tfactors11[0] = tfactors01

                tcore1 = multi_mode_dot(tl.tensor(tdata1), tfactors11, modes=modes, transpose=True)
                output1, score1 = Model11(timeseries1)
                td1 = torch.from_numpy(tcore1.reshape(tcore1.shape[0], -1)).float()
                output = model1(people1, td1, g, output1, score1)
                batch_loss = criterion1(output, tlabel1.long())
                _, test_pred = torch.max(output, 1)
                test_acc1 += (test_pred.cpu() == tlabel1.cpu()).sum().item()
                test_loss1 += batch_loss.item()

            for _, x in enumerate(dataloader2):
                tdata2 = x['pc2'].float()
                people2 = x['people2'].float()
                timeseries2 = x['timeseries2'].float()
                tlabel2 = x['label2'].float()
                tfactors02 = np.identity(tdata2.size(0))
                tfactors22 = factors22
                tfactors22[0] = tfactors02

                tcore2 = multi_mode_dot(tl.tensor(tdata2), tfactors22, modes=modes, transpose=True)
                output2, score2 = Model22(timeseries2)
                td2 = torch.from_numpy(tcore2.reshape(tcore2.shape[0], -1)).float()
                output = model2(people2, td2, g, output2, score2)
                batch_loss2 = criterion2(output, tlabel2.long())
                _, test_pred = torch.max(output, 1)
                test_acc2 += (test_pred.cpu() == tlabel2.cpu()).sum().item()
                test_loss2 += batch_loss2.item()

            for _, x in enumerate(dataloader3):
                tdata3 = x['pc3'].float()
                people3 = x['people3'].float()
                timeseries3 = x['timeseries3'].float()
                tlabel3 = x['label3'].float()
                tfactors03 = np.identity(tdata3.size(0))
                tfactors33 = factors33
                tfactors33[0] = tfactors03

                tcore3 = multi_mode_dot(tl.tensor(tdata3), tfactors33, modes=modes, transpose=True)
                output3, score3 = Model33(timeseries3)
                td3 = torch.from_numpy(tcore3.reshape(tcore3.shape[0], -1)).float()
                output = model3(people3, td3, g, output3, score3)
                batch_loss3 = criterion3(output, tlabel3.long())
                _, test_pred = torch.max(output, 1)
                test_acc3 += (test_pred.cpu() == tlabel3.cpu()).sum().item()
                test_loss3 += batch_loss3.item()
            if epoch == n_epoch-1:
                test_acc11.append(test_acc1 / tdata1.shape[0])
                test_acc22.append(test_acc2 / tdata2.shape[0])
                test_acc33.append(test_acc3 / tdata3.shape[0])


            print('NYUk_fold: {:03d} [{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f} | Test Acc: {:3.6f} loss: {:3.6f}'.format(
                k+1, epoch + 1, 50, train_acc1 / data1.shape[0], train_loss1,
                test_acc1 / tdata1.shape[0], test_loss1
            ))
            print(
                'UMk_fold: {:03d} [{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f} | Test Acc: {:3.6f} loss: {:3.6f}'.format(
                    k + 1, epoch + 1, 50, train_acc2 / data2.shape[0], train_loss2,
                    test_acc2 / tdata2.shape[0], test_loss2
                ))
            print(
                'UCLAk_fold: {:03d} [{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f} | Test Acc: {:3.6f} loss: {:3.6f}'.format(
                    k + 1, epoch + 1, 50, train_acc3 / data3.shape[0], train_loss3,
                    test_acc3 / tdata3.shape[0], test_loss3
                ))
    parameter_init1 = parameter_init_fix1
    parameter_init2 = parameter_init_fix2
    parameter_init3 = parameter_init_fix3
print('NYU Acc: {:3.6f} UM ACC: {:3.6f} UCLA Acc: {:3.6f}'.format(
                    sum(test_acc11)/5, sum(test_acc22)/5, sum(test_acc33)/5
                ))
print('end')


















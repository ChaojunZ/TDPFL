import numpy as np
from tensorly.base import unfold
from tensorly.tenalg import mode_dot, multi_mode_dot
from sklearn.decomposition import SparseCoder
from sklearn.metrics import mutual_info_score
from sklearn.metrics.pairwise import rbf_kernel
import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf
def K_Sparsity(matrix, m):
    num_subj, num_node, _ = matrix.shape
    threshold = np.percentile(matrix, 100 - m, axis=(1, 2), keepdims=True)
    threshold_matrix = np.repeat(threshold, num_node, axis=2)
    threshold_matrix = np.repeat(threshold_matrix, num_node, axis=1)
    r = np.where(matrix < threshold_matrix, 0, matrix)
    return r


def PC(data,Sp_Ratio=50):
    data = np.transpose(data, (0, 2, 1))
    Network = []
    for i in range(len(data)):
        signal = data[i].T
        network = np.corrcoef(signal)
        Network.append(network)
    PC_matrices = np.array(Network)
    PC_matrices = K_Sparsity(PC_matrices, Sp_Ratio)
    PC_matrices[np.isnan(PC_matrices)] = 0  # 将所有nan替换为0
    PC_matrices[np.isinf(PC_matrices)] = 0  # 将所有inf替换为0
    return PC_matrices

def SR(data, lam=0.01, Sp_Ratio=50):
    data = np.transpose(data, (0, 2, 1))
    Network = []
    for i in range(len(data)):
        signal = data[i].T
        sparse_coder = SparseCoder(dictionary=signal, transform_algorithm='lasso_lars', transform_alpha=lam)
        network = sparse_coder.transform(signal)
        network = (network + network.T) / 2
        Network.append(network)
    SR_matrices = np.array(Network)
    SR_matrices = K_Sparsity(SR_matrices, Sp_Ratio)
    SR_matrices[np.isnan(SR_matrices)] = 0  # 将所有nan替换为0
    SR_matrices[np.isinf(SR_matrices)] = 0  # 将所有inf替换为0
    return SR_matrices




def HOFC(data,Sp_Ratio=50):
    data = np.transpose(data, (0, 2, 1))
    Network = []
    for i in range(len(data)):
        signal = data[i].T
        network = np.corrcoef(np.corrcoef(signal))
        Network.append(network)
    HOFC_matrices = np.array(Network)
    HOFC_matrices = K_Sparsity(HOFC_matrices, Sp_Ratio)
    HOFC_matrices[np.isnan(HOFC_matrices)] = 0  # 将所有nan替换为0
    HOFC_matrices[np.isinf(HOFC_matrices)] = 0  # 将所有inf替换为0
    return HOFC_matrices

def MI(data, Sp_Ratio=50):
    data = np.transpose(data, (0, 2, 1))
    Network = []
    for i in range(len(data)):
        signal = data[i]
        network = mutual_information_matrix(signal)
        Network.append(network)
    MI_matrices = np.array(Network)
    MI_matrices = K_Sparsity(MI_matrices, Sp_Ratio)
    return MI_matrices

def mutual_information_matrix(data):
    num_features = data.shape[1]
    mi_matrix = np.zeros((num_features, num_features))
    for i in range(num_features):
        for j in range(num_features):
            mi_matrix[i, j] = mutual_info_score(data[:, i], data[:, j])
    return mi_matrix

# def FBN(data):
#     PC = PC(data, 80)
#     SR = SR(data, 0.1, 80)
#     MI = MI(data, 80)
#     HOFC = HOFC(data, 80)
#     tensors = tf.stack([PC, SR, MI, HOFC], axis=1)
#     return tensors


# 计算因子矩阵
def getUS(X, factors, index):
    if index == 1:  # 省略张量与factors[1]的模乘
        core = mode_dot(X, factors[0].T, 0)
        core1 = mode_dot(core, factors[2].T, 2)
        core_app = mode_dot(core1, factors[3].T, 3)

    elif index == 2:  # 省略张量与factors[2]的模乘
        core = mode_dot(X, factors[0].T, 0)
        core1 = mode_dot(core, factors[1].T, 1)
        core_app = mode_dot(core1, factors[3].T, 3)

    elif index == 3:
        core = mode_dot(X, factors[0].T, 0)
        core1 = mode_dot(core, factors[1].T, 1)
        core_app = mode_dot(core1, factors[2].T, 2)

    eigv, s, _ = np.linalg.svd(unfold(core_app, index))
    s = np.diag(s)
    US = np.dot(eigv, s)
    return US


def compute_cost_matrix(A, B):
    m, n = A.shape
    cost_matrix = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            # cost_matrix[i, j] = np.linalg.norm(A[i] - B[j])  # 欧氏距离
            # 或者使用余弦相似度
            from scipy.spatial import distance
            cost_matrix[i, j] = 1 - distance.cosine(A[i], B[j])
    return cost_matrix

def map_source_domain(A, Pi, B):
    m, n = A.shape
    mapped_A = np.zeros((m, n))
    for i in range(m):
        mapped_A[i] = np.sum([Pi[i, j] * B[j] for j in range(m)], axis=0)
    return mapped_A



class Population(nn.Module):
    def __init__(self, p_dim, bold_dim, hidden_dim, latent_dim, output_dim, Type, out_dim):
        super(Population, self).__init__()
        self.fc_p = nn.Linear(p_dim, output_dim)
        self.fc_latent = nn.Linear(latent_dim, output_dim)
        self.fc_final = nn.Linear(output_dim+hidden_dim, out_dim)

    def forward(self, p, latent, g):
        # p: demographic information, bold: flattened bold upper triangular matrix, latent: logit from STAGIN, g: gamma
        p = self.fc_p(p)
        f_o = torch.cat((p, self.fc_latent(latent)),dim=1)
        out = self.fc_final(f_o)
        return out, f_o

class STAR1(nn.Module):
    def __init__(self, d_series, d_core):
        super(STAR1, self).__init__()
        """
        STar Aggregate-Redistribute Module
        """

        self.gen1 = nn.Linear(d_series, d_series)
        self.gen2 = nn.Linear(d_series, d_core)
        self.gen3 = nn.Linear(d_series + d_core, d_series)
        self.gen4 = nn.Linear(d_series, d_series)

    def forward(self, input, *args, **kwargs):
        batch_size, channels, d_series = input.shape  # (B,D,L)

        # set FFN
        combined_mean = F.gelu(self.gen1(input))  # (B,D,L)-->(B,D,L)
        combined_mean = self.gen2(combined_mean)  # (B,D,L)-->(B,D,L_core)

        # stochastic pooling
        if self.training:
            ratio = F.softmax(combined_mean, dim=1)  # 在通道方向上执行softmax,为随机池化生成一个概率权重: (B,D,L_core)-->(B,D,L_core)
            ratio = ratio.permute(0, 2, 1)  # (B,D,L_core)--permute->(B,L_core,D)
            ratio = ratio.reshape(-1, channels)  # 转换为2维, 便于进行采样: (B,L_core,D)--reshape-->(B*L_core,D)
            indices = torch.multinomial(ratio,
                                        1)  # 从多项分布ratio的每一行中抽取一个样本,返回值是采样得到的类别的索引: (B*L_core,1); 输入如果是一维张量,它表示每个类别的概率;如果是二维张量,每行表示一个概率分布
            indices = indices.view(batch_size, -1, 1).permute(0, 2,
                                                              1)  # (B*L_core,1)--view--> (B,L_core,1)--permute-->(B,1,L_core)
            combined_mean = torch.gather(combined_mean, 1,
                                         indices)  # 根据索引indices在D方向上选择对应的通道元素(理解为:选择重要的通道信息): (B,D,L_core)--gather-->(B,1,L_core)    # gather函数不了解的看这个:https://zhuanlan.zhihu.com/p/661293803
            core = np.squeeze(combined_mean, axis=1)
            combined_mean = combined_mean.repeat(1, channels,
                                                 1)  # 复制D份,将随机选择的core表示应用到所有通道上: (B,1,L_core)--repeat-->(B,D,L_core)
        else:
            weight = F.softmax(combined_mean, dim=1)  # 处于非训练模式时, 首先通过softmax生成一个权重分布:(B,D,L_core)-->(B,D,L_core)
            combined_mean = torch.sum(combined_mean * weight, dim=1, keepdim=True)
            core = np.squeeze(combined_mean, axis=1)
            combined_mean = combined_mean.repeat(1, channels, 1)  # 直接在D方向上进行加权求和, 然后复制D份: (B,D,L_core)--sum-->(B,1,L_core)--repeat-->(B,D,L_core)

        # mlp fusion
        combined_mean_cat = torch.cat([input, combined_mean], -1)  # (B,D,L)--cat--(B,D,L_core)==(B,D,L+L_core)
        combined_mean_cat = F.gelu(self.gen3(combined_mean_cat))  # (B,D,L+L_core)-->(B,D,L)
        combined_mean_cat = self.gen4(combined_mean_cat)  # (B,D,L)-->(B,D,L)
        output = combined_mean_cat

        return output, core

class STAR2(nn.Module):
    def __init__(self, d_series, d_core):
        super(STAR2, self).__init__()
        """
        STar Aggregate-Redistribute Module
        """

        self.gen1 = nn.Linear(d_series, d_series)
        self.gen2 = nn.Linear(d_series, d_core)
        self.gen3 = nn.Linear(d_series + d_core, d_series)
        self.gen4 = nn.Linear(d_series, d_series)

    def forward(self, input, *args, **kwargs):
        batch_size, channels, d_series = input.shape  # (B,D,L)

        # set FFN
        combined_mean = F.gelu(self.gen1(input))  # (B,D,L)-->(B,D,L)
        combined_mean = self.gen2(combined_mean)  # (B,D,L)-->(B,D,L_core)

        # stochastic pooling
        if self.training:
            ratio = F.softmax(combined_mean, dim=1)  # 在通道方向上执行softmax,为随机池化生成一个概率权重: (B,D,L_core)-->(B,D,L_core)
            ratio = ratio.permute(0, 2, 1)  # (B,D,L_core)--permute->(B,L_core,D)
            ratio = ratio.reshape(-1, channels)  # 转换为2维, 便于进行采样: (B,L_core,D)--reshape-->(B*L_core,D)
            indices = torch.multinomial(ratio,
                                        1)  # 从多项分布ratio的每一行中抽取一个样本,返回值是采样得到的类别的索引: (B*L_core,1); 输入如果是一维张量,它表示每个类别的概率;如果是二维张量,每行表示一个概率分布
            indices = indices.view(batch_size, -1, 1).permute(0, 2,
                                                              1)  # (B*L_core,1)--view--> (B,L_core,1)--permute-->(B,1,L_core)
            combined_mean = torch.gather(combined_mean, 1,
                                         indices)  # 根据索引indices在D方向上选择对应的通道元素(理解为:选择重要的通道信息): (B,D,L_core)--gather-->(B,1,L_core)    # gather函数不了解的看这个:https://zhuanlan.zhihu.com/p/661293803
            core = np.squeeze(combined_mean, axis=1)
            combined_mean = combined_mean.repeat(1, channels,
                                                 1)  # 复制D份,将随机选择的core表示应用到所有通道上: (B,1,L_core)--repeat-->(B,D,L_core)
        else:
            weight = F.softmax(combined_mean, dim=1)  # 处于非训练模式时, 首先通过softmax生成一个权重分布:(B,D,L_core)-->(B,D,L_core)
            combined_mean = torch.sum(combined_mean * weight, dim=1, keepdim=True)
            core = np.squeeze(combined_mean, axis=1)
            combined_mean = combined_mean.repeat(1, channels, 1)  # 直接在D方向上进行加权求和, 然后复制D份: (B,D,L_core)--sum-->(B,1,L_core)--repeat-->(B,D,L_core)

        # mlp fusion
        combined_mean_cat = torch.cat([input, combined_mean], -1)  # (B,D,L)--cat--(B,D,L_core)==(B,D,L+L_core)
        combined_mean_cat = F.gelu(self.gen3(combined_mean_cat))  # (B,D,L+L_core)-->(B,D,L)
        combined_mean_cat = self.gen4(combined_mean_cat)  # (B,D,L)-->(B,D,L)
        output = combined_mean_cat

        return output, core

class STAR3(nn.Module):
    def __init__(self, d_series, d_core):
        super(STAR3, self).__init__()
        """
        STar Aggregate-Redistribute Module
        """

        self.gen1 = nn.Linear(d_series, d_series)
        self.gen2 = nn.Linear(d_series, d_core)
        self.gen3 = nn.Linear(d_series + d_core, d_series)
        self.gen4 = nn.Linear(d_series, d_series)

    def forward(self, input, *args, **kwargs):
        batch_size, channels, d_series = input.shape  # (B,D,L)

        # set FFN
        combined_mean = F.gelu(self.gen1(input))  # (B,D,L)-->(B,D,L)
        combined_mean = self.gen2(combined_mean)  # (B,D,L)-->(B,D,L_core)

        # stochastic pooling
        if self.training:
            ratio = F.softmax(combined_mean, dim=1)  # 在通道方向上执行softmax,为随机池化生成一个概率权重: (B,D,L_core)-->(B,D,L_core)
            ratio = ratio.permute(0, 2, 1)  # (B,D,L_core)--permute->(B,L_core,D)
            ratio = ratio.reshape(-1, channels)  # 转换为2维, 便于进行采样: (B,L_core,D)--reshape-->(B*L_core,D)
            indices = torch.multinomial(ratio,
                                        1)  # 从多项分布ratio的每一行中抽取一个样本,返回值是采样得到的类别的索引: (B*L_core,1); 输入如果是一维张量,它表示每个类别的概率;如果是二维张量,每行表示一个概率分布
            indices = indices.view(batch_size, -1, 1).permute(0, 2,
                                                              1)  # (B*L_core,1)--view--> (B,L_core,1)--permute-->(B,1,L_core)
            combined_mean = torch.gather(combined_mean, 1,
                                         indices)  # 根据索引indices在D方向上选择对应的通道元素(理解为:选择重要的通道信息): (B,D,L_core)--gather-->(B,1,L_core)    # gather函数不了解的看这个:https://zhuanlan.zhihu.com/p/661293803
            core = np.squeeze(combined_mean, axis=1)
            combined_mean = combined_mean.repeat(1, channels,
                                                 1)  # 复制D份,将随机选择的core表示应用到所有通道上: (B,1,L_core)--repeat-->(B,D,L_core)
        else:
            weight = F.softmax(combined_mean, dim=1)  # 处于非训练模式时, 首先通过softmax生成一个权重分布:(B,D,L_core)-->(B,D,L_core)
            combined_mean = torch.sum(combined_mean * weight, dim=1, keepdim=True)
            core = np.squeeze(combined_mean, axis=1)
            combined_mean = combined_mean.repeat(1, channels, 1)  # 直接在D方向上进行加权求和, 然后复制D份: (B,D,L_core)--sum-->(B,1,L_core)--repeat-->(B,D,L_core)

        # mlp fusion
        combined_mean_cat = torch.cat([input, combined_mean], -1)  # (B,D,L)--cat--(B,D,L_core)==(B,D,L+L_core)
        combined_mean_cat = F.gelu(self.gen3(combined_mean_cat))  # (B,D,L+L_core)-->(B,D,L)
        combined_mean_cat = self.gen4(combined_mean_cat)  # (B,D,L)-->(B,D,L)
        output = combined_mean_cat

        return output, core


import collections

def W_A(para_list,data_num_list):
    # weighted average
    # para_list: parameters of local models
    # data_num_list: sample number of local sites
    dict=collections.OrderedDict()
    rate=[x/sum(data_num_list) for x in data_num_list]

    for i in range(len(para_list)):
        for key in para_list[i].keys():
            para_list[i][key]=para_list[i][key]*rate[i]

    for i in range(len(para_list)):
        if i==0:
            dict=para_list[i]
        else:
            for n in dict.keys():
                dict[n]=dict[n]+para_list[i][n]

    return dict



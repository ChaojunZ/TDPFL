from torch.utils.data import Dataset
import scipy.io
from sklearn.model_selection import StratifiedKFold
from fc import *
from torch import tensor, float32
from random import shuffle
import pandas as pd
from sklearn import preprocessing
# site1
class DatasetHCPRest1(Dataset):
    def __init__(self, k_fold=None):
        super().__init__()
        site1 = scipy.io.loadmat(r'D:\FLcode\data\site20.mat')
        site1_s = scipy.io.loadmat(r'D:\FLcode\data\site20_s.mat')
        f = pd.read_csv(r'D:\FLcode\data\site20_info.csv')
        bold1 = site1['site20DATA']
        s1 = site1_s['site20_s']
        #s1 = s1[:, :, :175]
        bold1[np.isnan(bold1)] = 0  # 
        bold1[np.isinf(bold1)] = 0  # 
        # min_val = np.min(bold1, axis=(-2, -1), keepdims=True)  # 
        # max_val = np.max(bold1, axis=(-2, -1), keepdims=True)  # 
        # range_val = max_val - min_val  #
        #
        # # 
        # range_val[range_val == 0] = 1
        #
        # bold1 = ((bold1 - min_val) / range_val) * 2 - 1  # 归一化到 (-1, 1)
        # X1 = PC(bold1)
        # X2 = SR(bold1)
        # X3 = HOFC(bold1)
        # tensors = tf.stack([X1, X2, X3], axis=1)
        tensors = np.array(bold1)
        sample = len(tensors)
        numbers = [int(x) for x in range(sample)]
        d1 = zip(numbers, tensors)
        s1 = zip(numbers, s1)

        people = f.to_numpy()[:, 2:].astype(np.float32)
        people = torch.from_numpy(preprocessing.StandardScaler().fit_transform(people))
        people = zip(numbers, people)
        self.pc1_dict = dict(d1)
        self.s1 = dict(s1)
        self.p = dict(people)
        self.full_subject_list = list(self.pc1_dict.keys())
        if k_fold is None:
            self.subject_list = self.full_subject_list
        else:
            self.k_fold = StratifiedKFold(k_fold, shuffle=True, random_state=0) if k_fold is not None else None
            self.k = None

        y = site1['site20_lab']
        y = np.squeeze(y)
        y = y.tolist()
        dy = zip(numbers, y)
        self.behavioral_dict = dict(dy)

        self.full_label_list = [self.behavioral_dict[int(subject)] for subject in self.full_subject_list]

    def __len__(self):
        return len(self.subject_list) if self.k is not None else len(self.full_subject_list)

    def set_fold(self, fold, train=True):
        assert self.k_fold is not None
        self.k = fold
        train_idx, test_idx = list(self.k_fold.split(self.full_subject_list, self.full_label_list))[fold]

        if train: shuffle(train_idx)
        self.subject_list = [self.full_subject_list[idx] for idx in train_idx] if train else [
            self.full_subject_list[idx] for idx in test_idx]

    def __getitem__(self, idx):
        subject = self.subject_list[idx]
        pc1 = self.pc1_dict[subject]
        s1 = self.s1[subject]
        people = self.p[subject]
        label = self.behavioral_dict[int(subject)]

        if label == 1:
            label = tensor(1)
        elif label == 0:
            label = tensor(0)
        else:
            raise
        return {'id': subject, 'pc1': tensor(pc1, dtype=float32), 'timeseries1': s1, 'label1': label, 'people1': people}

# site2
class DatasetHCPRest2(Dataset):
    def __init__(self, k_fold=None):
        super().__init__()
        site1 = scipy.io.loadmat(r'D:\FLcode\data\site21.mat')
        site1_s = scipy.io.loadmat(r'D:\FLcode\data\site21_s.mat')
        f = pd.read_csv(r'D:\FLcode\data\site21_info.csv')
        bold1 = site1['site21DATA']
        s1 = site1_s['site21_s']
        #s1 = s1[:, :, :175]
        bold1[np.isnan(bold1)] = 0  # 
        bold1[np.isinf(bold1)] = 0  #
        # min_val = np.min(bold1, axis=(-2, -1), keepdims=True)  # 
        # max_val = np.max(bold1, axis=(-2, -1), keepdims=True)  #
        # range_val = max_val - min_val  # 
        #
        # 
        # range_val[range_val == 0] = 1
        #
        # bold1 = ((bold1 - min_val) / range_val) * 2 - 1  # 归一化到 (-1, 1)
        # X1 = PC(bold1)
        # X2 = SR(bold1)
        # X3 = HOFC(bold1)
        # tensors = tf.stack([X1, X2, X3], axis=1)
        tensors = np.array(bold1)
        sample = len(tensors)
        numbers = [int(x) for x in range(sample)]
        d1 = zip(numbers, tensors)
        s1 = zip(numbers, s1)

        people = f.to_numpy()[:, 2:].astype(np.float32)
        people = torch.from_numpy(preprocessing.StandardScaler().fit_transform(people))
        people = zip(numbers, people)
        self.pc1_dict = dict(d1)
        self.s1 = dict(s1)
        self.p = dict(people)
        self.full_subject_list = list(self.pc1_dict.keys())
        if k_fold is None:
            self.subject_list = self.full_subject_list
        else:
            self.k_fold = StratifiedKFold(k_fold, shuffle=True, random_state=0) if k_fold is not None else None
            self.k = None

        y = site1['site21_lab']
        y = np.squeeze(y)
        y = y.tolist()
        dy = zip(numbers, y)
        self.behavioral_dict = dict(dy)

        self.full_label_list = [self.behavioral_dict[int(subject)] for subject in self.full_subject_list]

    def __len__(self):
        return len(self.subject_list) if self.k is not None else len(self.full_subject_list)

    def set_fold(self, fold, train=True):
        assert self.k_fold is not None
        self.k = fold
        train_idx, test_idx = list(self.k_fold.split(self.full_subject_list, self.full_label_list))[fold]

        if train: shuffle(train_idx)
        self.subject_list = [self.full_subject_list[idx] for idx in train_idx] if train else [
            self.full_subject_list[idx] for idx in test_idx]

    def __getitem__(self, idx):
        subject = self.subject_list[idx]
        pc1 = self.pc1_dict[subject]
        s1 = self.s1[subject]
        people = self.p[subject]
        label = self.behavioral_dict[int(subject)]

        if label == 1:
            label = tensor(1)
        elif label == 0:
            label = tensor(0)
        else:
            raise
        return {'id': subject, 'pc2': tensor(pc1, dtype=float32), 'timeseries2': s1, 'label2': label, 'people2': people}

# site3
class DatasetHCPRest3(Dataset):
    def __init__(self, k_fold=None):
        super().__init__()
        site1 = scipy.io.loadmat(r'D:\FLcode\data\site25.mat')
        site1_s = scipy.io.loadmat(r'D:\FLcode\data\site25_s.mat')
        f = pd.read_csv(r'D:\FLcode\data\site25_info.csv')
        bold1 = site1['site25DATA']
        s1 = site1_s['site25_s']
        #s1 = s1[:, :, :175]
        # site1 = scipy.io.loadmat(r'D:\FLcode\data\UCLA.mat')
        # f = pd.read_csv(r'D:\FLcode\data\UCLA_info.csv')
        # bold1 = site1['UCLADATA']
        bold1[np.isnan(bold1)] = 0  
        bold1[np.isinf(bold1)] = 0 
        # min_val = np.min(bold1, axis=(-2, -1), keepdims=True)  
        # max_val = np.max(bold1, axis=(-2, -1), keepdims=True) 
        # range_val = max_val - min_val  # 计算值域
        #
        
        # range_val[range_val == 0] = 1
        #
        # bold1 = ((bold1 - min_val) / range_val) * 2 - 1  # 归一化到 (-1, 1)
        # X1 = PC(bold1)
        # X2 = SR(bold1)
        # X3 = HOFC(bold1)
        # tensors = tf.stack([X1, X2, X3], axis=1)
        tensors = np.array(bold1)
        sample = len(tensors)
        numbers = [int(x) for x in range(sample)]
        d1 = zip(numbers, tensors)
        s1 = zip(numbers, s1)

        people = f.to_numpy()[:, 2:].astype(np.float32)
        people = torch.from_numpy(preprocessing.StandardScaler().fit_transform(people))
        people = zip(numbers, people)
        self.pc1_dict = dict(d1)
        self.s1 = dict(s1)
        self.p = dict(people)
        self.full_subject_list = list(self.pc1_dict.keys())
        if k_fold is None:
            self.subject_list = self.full_subject_list
        else:
            self.k_fold = StratifiedKFold(k_fold, shuffle=True, random_state=0) if k_fold is not None else None
            self.k = None

        y = site1['site25_lab']
        # y = site1['ucla_lab']
        y = np.squeeze(y)
        y = y.tolist()
        dy = zip(numbers, y)
        self.behavioral_dict = dict(dy)

        self.full_label_list = [self.behavioral_dict[int(subject)] for subject in self.full_subject_list]

    def __len__(self):
        return len(self.subject_list) if self.k is not None else len(self.full_subject_list)

    def set_fold(self, fold, train=True):
        assert self.k_fold is not None
        self.k = fold
        train_idx, test_idx = list(self.k_fold.split(self.full_subject_list, self.full_label_list))[fold]

        if train: shuffle(train_idx)
        self.subject_list = [self.full_subject_list[idx] for idx in train_idx] if train else [
            self.full_subject_list[idx] for idx in test_idx]

    def __getitem__(self, idx):
        subject = self.subject_list[idx]
        pc1 = self.pc1_dict[subject]
        s1 = self.s1[subject]
        people = self.p[subject]
        label = self.behavioral_dict[int(subject)]

        if label == 1:
            label = tensor(1)
        elif label == 0:
            label = tensor(0)
        else:
            raise
        return {'id': subject, 'pc3': tensor(pc1, dtype=float32), 'timeseries3': s1, 'label3': label, 'people3': people}

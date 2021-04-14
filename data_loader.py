import numpy as np
from scipy import stats
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from datasets import bcic4_2a
from utils import plv_tensor, corr_tensor, normalize_adj_tensor, segment_tensor, transpose_tensor

import scipy.io as sio

class CustomDataset(Dataset):

    def __init__(self, data):
        super(CustomDataset, self).__init__()

        X = data[0]
        y = data[1]

        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def data_loader(sub_ind, batch_size, CSP):
    print("[Loading data...]")

    # direction of data

    base = 'C:/Users/MAI Lab/JupyterProjects'
    folder = '/morlet_cnn(0-42)'
    sub = '/sub'
    i = '%s' %(sub_ind)

    if CSP == True :
        train = '/csp_train_X.mat'
        test = '/csp_test_X.mat'

    else :
        train = '/session_train_X.mat'
        test = '/session_test_X.mat'

    train_y = '/session_train_y.mat'
    test_y = '/session_test_y.mat'

    train_dir = base+folder+sub+i+train
    test_dir = base+folder+sub+i+test
    train_y_dir = base+folder+sub+i+train_y
    test_y_dir = base+folder+sub+i+test_y


    # Load train data

    if CSP == True :
        train_X = sio.loadmat(train_dir)['csp_train_X']

    else :
        train_X = sio.loadmat(train_dir)['session_train_X']

    train_y = sio.loadmat(train_y_dir)['session_train_y']

    train_X = np.squeeze(train_X)
    train_y = np.squeeze(train_y)

    data = [train_X, train_y]
    trainset = CustomDataset(data)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)

    # Load test data

    if CSP == True :
        test_X = sio.loadmat(test_dir)['csp_test_X']

    else :
        test_X = sio.loadmat(test_dir)['session_test_X']

    test_y = sio.loadmat(test_y_dir)['session_test_y']
    test_X = np.squeeze(test_X)
    test_y = np.squeeze(test_y)

    data = [test_X, test_y]
    testset = CustomDataset(data)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0)

    # Print
    print(f"train_set size: {train_loader.dataset.X.shape}")
    print(f"val_set size: {test_loader.dataset.X.shape}")
    return train_loader, test_loader

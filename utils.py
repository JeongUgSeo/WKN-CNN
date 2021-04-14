import os
import sys
import json
import time
import itertools
import h5py

import numpy as np
import scipy.signal as sig

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F

# wave kernel net에서 필요한거
import torch.utils.model_zoo as model_zoo
from math import pi

np.set_printoptions(linewidth=np.inf)


# Print
def print_off():
    sys.stdout = open(os.devnull, 'w')


def print_on():
    sys.stdout = sys.__stdout__


def print_update(sentence, i):
    """

    Args:
        sentence: sentence you want
        i: index in for loop

    Returns:

    """

    print(sentence, end='') if i == 0 else print('\r' + sentence, end='')


def print_dict(dictionary):
    for key, value in dictionary.items():
        print(f"{key}: {value}")
    print("")


def print_info(args):
    print("")
    print(f"PID: {os.getpid()}\n")
    print(f"Python version: {sys.version.split(' ')[0]}")
    print(f"Pytorch version: {torch.__version__}")
    print("")
    print_dict(args)


# Time
def convert_time(sec):
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    print(f"Total time: {h:02}:{m:02}:{s:02}")


def timeit(func):
    start = time.time()

    def decorator():
        _return = func()
        convert_time(time.time() - start)
        return _return

    return decorator


# Handling file an directory
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Creating directory. " + directory)


def write_json(path, json_data):
    with open(path, "w") as json_file:
        json.dump(json_data, json_file)


def read_json(path):
    with open(path, "r") as json_file:
        file = json.load(json_file)
    return file


def h5py_read(path):
    with h5py.File(path, 'r') as file:
        data = file['data'][()]
        return data


def make_save_path(path):
    if os.path.exists(path):
        return os.path.join(path, str(len(os.listdir(path))))
    else:
        return os.path.join(path, "0")


# Handling array
def order_change(array, order):
    array = list(array)
    tmp = array[order[0]]
    array[order[0]] = array[order[1]]
    array[order[1]] = tmp
    return array


def array_equal(A, B):
    return np.array_equal(np.round(A, 5), np.round(B, 5))


def convert_list(string):
    lst = string.split(",")
    assert len(lst) % 2 == 0, "Length of the list must be even number."
    it = iter(lst)
    return [list(map(int, itertools.islice(it, i))) for i in ([2] * (len(lst) // 2))]


def str2list_int(string):
    if string == 'all':
        return 'all'
    else:
        return list(map(int, string.split(",")))


def str2list(string):
    if string == 'all':
        return 'all'
    else:
        return string.split(",")


def str2dict(string):
    lst = string.split("_")
    return {key: str2list_int(value) for key, value in zip(lst[::2], lst[1::2])}


# Operation
def plv_signal(sig1, sig2):
    sig1_hill = sig.hilbert(sig1)
    sig2_hill = sig.hilbert(sig2)
    phase_1 = np.angle(sig1_hill)
    phase_2 = np.angle(sig2_hill)
    phase_diff = phase_1 - phase_2
    _plv = np.abs(np.mean([np.exp(complex(0, phase)) for phase in phase_diff]))
    return _plv


# Tensor operation
def transpose_tensor(tensor, order):
    return np.transpose(tensor, order_change(np.arange(len(tensor.shape)), order))


def plv_tensor(tensor):
    """

    Parameters
    ----------
    tensor: [..., channels, times]

    Returns
    -------

    """
    tensor = np.angle(sig.hilbert(tensor))
    tensor = np.exp(tensor * 1j)
    _plv = np.abs(
        (tensor @ (np.transpose(tensor, order_change(np.arange(len(tensor.shape)), [-1, -2])) ** -1)) / np.size(tensor,
                                                                                                                -1))
    return _plv


def corr_tensor(tensor):
    """

    Parameters
    ----------
    tensor: [..., channels, times]

    Returns
    -------

    """
    mean = tensor.mean(axis=-1, keepdims=True)
    tensor2 = tensor - mean
    tensor3 = tensor2 @ np.transpose(tensor2, order_change(np.arange(len(tensor2.shape)), [-1, -2]))
    tensor4 = np.sqrt(np.expand_dims(np.diagonal(tensor3, axis1=-2, axis2=-1), axis=-1) @ np.expand_dims(
        np.diagonal(tensor3, axis1=-2, axis2=-1), axis=-2))
    corr = tensor3 / tensor4
    return corr


def normalize_adj_tensor(adj):
    diag = np.power(adj.sum(-2, keepdims=True), -0.5)
    diag[np.isinf(diag)] = 0.
    return transpose_tensor((adj * diag), [-1, -2]) * diag


def segment_tensor(tensor, window_size, step):
    """

    Parameters
    ----------
    tensor: [..., chans, times]
    window_size
    step

    Returns: [shape[0], segment, ..., chans, times]
    -------

    """
    segment = []
    times = np.arange(tensor.shape[-1])
    start = times[::step]
    end = start + window_size
    for s, e in zip(start, end):
        if e > len(times):
            break
        segment.append(tensor[..., s:e])
    segment = transpose_tensor(np.array(segment), [0, 1])
    return segment


# Miscellaneous
def control_random(seed, gpu):
    # Control randomness
    np.random.seed(seed)
    torch.manual_seed(seed)
    if gpu == "multi":
        torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    else:
        torch.cuda.manual_seed(seed)
    cudnn.benchmark = False  # If you want set randomness, cudnn.benchmark = False
    cudnn.deterministic = True
    print(f"Control randomness - seed {seed}")


def Laplace(p):
    A = 0.08
    ep = 0.03
    tal = 0.1
    f = 50
    w = 2 * pi * f
    q = torch.tensor(1 - pow(ep, 2))
    y = A * torch.exp((-ep / (torch.sqrt(q))) * (w * (p - tal))) * (-torch.sin(w * (p - tal)))

    return y


class Laplace_fast(nn.Module):

    def __init__(self, out_channels, kernel_size, in_channels=1):
        super(Laplace_fast, self).__init__()

        #         if in_channels != 1:

        #             msg = "MexhConv only support one input channel (here, in_channels = {%i})" % (in_channels)
        #             raise ValueError(msg)

        self.out_channels = out_channels
        self.kernel_size = kernel_size - 1

        if kernel_size % 2 == 0:
            self.kernel_size = self.kernel_size + 1

        self.a_ = nn.Parameter(torch.linspace(1, 10, out_channels)).view(-1, 1)

        self.b_ = nn.Parameter(torch.linspace(0, 10, out_channels)).view(-1, 1)

    def forward(self, waveforms):
        time_disc = torch.linspace(0, 1, steps=int((self.kernel_size)))

        p1 = time_disc.cuda() - self.b_.cuda() / self.a_.cuda()

        laplace_filter = Laplace(p1)

        self.filters = (laplace_filter).view(self.out_channels, 1,
                                             self.kernel_size).cuda()  # 원래 in_channels는 없었고 1만 되어 있었음

        return F.conv1d(waveforms, self.filters, stride=1, padding=1, dilation=1, bias=None, groups=1)


def Morlet(p):
    C = pow(pi, 0.25)
    # p = 0.03 * p
    y = C * torch.exp(-torch.pow(p, 2) / 2) * torch.cos(2 * pi * p) # wavelet을 만드는 부분
    return y


class Morlet_fast(nn.Module):

    def __init__(self, out_channels, kernel_size, in_channels=1):

        super(Morlet_fast, self).__init__()

        if in_channels != 1:
            msg = "MexhConv only support one input channel (here, in_channels = {%i})" % (in_channels)
            raise ValueError(msg)

        self.out_channels = out_channels
        self.kernel_size = kernel_size - 1

        if kernel_size % 2 == 0:
            self.kernel_size = self.kernel_size + 1

        self.a_ = nn.Parameter(torch.linspace(1, 10, out_channels)).view(-1, 1) # out_channels 개수 만큼 morlet wavelet을 만들겠다는 소리 / 여기 값들이 계속 학습됨

        self.b_ = nn.Parameter(torch.linspace(0, 10, out_channels)).view(-1, 1) #

    def forward(self, waveforms):

        time_disc_right = torch.linspace(0, (self.kernel_size / 2) - 1,
                                         steps=int((self.kernel_size / 2))) # kernel 왼쪽 부분의 파라미터값이 들어갈 간격

        time_disc_left = torch.linspace(-(self.kernel_size / 2) + 1, -1,
                                        steps=int((self.kernel_size / 2))) # kernel 오른쪽 부분의 파라미터값이 들어갈 간격

        p1 = time_disc_right.cuda() - self.b_.cuda() / self.a_.cuda()
        p2 = time_disc_left.cuda() - self.b_.cuda() / self.a_.cuda()

        Morlet_right = Morlet(p1)
        Morlet_left = Morlet(p2)

        Morlet_filter = torch.cat([Morlet_left, Morlet_right], dim=1)  # 40x1x250

        self.filters = (Morlet_filter).view(self.out_channels, 1, self.kernel_size).cuda()

        return F.conv1d(waveforms, self.filters, stride=1, padding=1, dilation=1, bias=None, groups=1)


def acc_list(epoc):

    acc = 0
    acc_ind = 0
    certain_epoc_list = []
    certain_epoc_ind = 0

    for j in range(0, 5000):

        x = 0
        x2 = []

        for i in range(1, 10):
            dir = 'sub' + f'{i}' + '_acc.npy'
            data = np.load(dir)
            x += data[j]
            x2.append(data[j])

            if j == epoc-1:

                certain_epoc_list.append(data[j])
                certain_epoc_ind = j+1


        if x/9 > acc :
            acc = x/9
            acc_ind = j+1
            acc_list = x2


    print('max_acc', acc)
    print('max_ind', acc_ind)
    print('acc_list', acc_list)
    print('certain_ind', certain_epoc_ind)
    print('acc_list2', certain_epoc_list)




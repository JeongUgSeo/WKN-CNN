import torch.nn as nn
import torch.nn.functional as F
from utils import Morlet_fast, Laplace_fast

# Basic module
class CNN(nn.Module):
    def __init__(self, shape):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(shape[1], 50, kernel_size=(3, 3), stride=1)
        self.bn1 = nn.BatchNorm2d(50)
        self.mp1 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv2 = nn.Conv2d(50, 50, kernel_size=(3, 3), stride=1)
        self.bn2 = nn.BatchNorm2d(50)
        self.mp2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv3 = nn.Conv2d(50, 50, kernel_size=(3, 3), stride=1)
        self.bn3 = nn.BatchNorm2d(50)
        self.mp3 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv4 = nn.Conv2d(50, 50, kernel_size=(3, 3), stride=1)
        self.bn4 = nn.BatchNorm2d(50)
        self.mp4 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv5 = nn.Conv2d(50, 50, kernel_size=(3, 3), stride=1)
        self.bn5 = nn.BatchNorm2d(50)
        self.mp5 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv6 = nn.Conv2d(50, 50, kernel_size=(1, 3), stride=1)
        self.bn6 = nn.BatchNorm2d(50)
        self.mp6 = nn.MaxPool2d(kernel_size=(1, 2))

        self.conv_module = nn.Sequential(
            self.conv1,
            self.bn1,
            nn.LeakyReLU(),
            self.mp1,
            self.conv2,
            self.bn2,
            nn.LeakyReLU(),
            self.mp2,

        )

        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(50*1*68, 2) #16-31 hz beta

        self.fc_module = nn.Sequential(
        self.flat,
        self.fc1
        )

    def forward(self, x):
        x = self.conv_module(x)
        x = self.fc_module(x)

        return F.log_softmax(x, dim=1)


class EEGNet2D(nn.Module):
    def __init__(self, shape, cfg):
        super(EEGNet2D, self).__init__()

        global filter_n
        filter_n = cfg[0]
        filter_size = cfg[1]
        ch_n = shape[1]

        self.laplace_conv = Laplace_fast(filter_n, filter_size, 1)
        self.bn_laplace = nn.BatchNorm1d(filter_n)
        self.mp_laplace = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.morlet_conv = Morlet_fast(filter_n, filter_size, 1)

        self.Conv_0 = nn.Sequential(
            nn.Conv2d(ch_n, 44, kernel_size=(1, 1), stride=(1, 2)),
            nn.BatchNorm2d(44),
            nn.LeakyReLU(),
            nn.MaxPool2d((1, 2))
        )
        self.Conv_1 = nn.Sequential(
            nn.Conv2d(44, 44, kernel_size=(3, 15), stride=(1, 2)),
            nn.BatchNorm2d(44),
            nn.LeakyReLU(),
            nn.MaxPool2d((2, 4))
        )
        self.Conv_2 = nn.Sequential(
            nn.Conv2d(44, 66, kernel_size=(2, 3), stride=(1, 2)),  # 원랜 커널 3,3
            nn.BatchNorm2d(66),
            nn.LeakyReLU(),
            nn.MaxPool2d((2, 2)),
        )
        self.Conv_3 = nn.Sequential(
            nn.Conv2d(66, 66, kernel_size=(2, 3), stride=(1, 1)),  # 원랜 커널 3,3
            nn.BatchNorm2d(66),
            nn.LeakyReLU(),
            nn.MaxPool2d((2, 2)),
        )
        self.classify = nn.Sequential(
            nn.Linear(in_features=2904, out_features=2)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        ch_n = x.shape[1]

        x = x.view(batch_size * ch_n, 1, -1)
        x = self.morlet_conv(x)
        # x = self.bn_laplace(x)
        # x = F.leaky_relu(x)
        # x = self.mp_laplace(x)
        x = x.view(batch_size, ch_n, filter_n, -1)  # 36,22,filter_n,time_n

        x = self.Conv_0(x)
        x = self.Conv_1(x)
        # x = self.Conv_2(x)
        #         x = self.Conv_3(x)
        x = x.view(x.shape[0], -1)
        global out_linear
        out_linear = x
        x = self.classify(x)

        return F.log_softmax(x, dim=1)

class Seo_CNN(nn.Module):
    def __init__(self, shape, cfg):
        super(Seo_CNN, self).__init__()

        global filter_n
        filter_n = cfg[0]
        filter_size = cfg[1]
        ch_n = shape[1]

        # self.laplace_conv = Laplace_fast(filter_n, filter_size, 1)
        # self.bn_laplace = nn.BatchNorm1d(filter_n)
        # self.mp_laplace = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.morlet_conv = Morlet_fast(filter_n, 16, 1)

        self.Conv_0 = nn.Sequential(
            nn.Conv2d(ch_n, 44, kernel_size=(6, 3), stride=(1, 1)),
            nn.BatchNorm2d(44),
            nn.LeakyReLU(),
            nn.MaxPool2d((1, 2))
        )
        self.Conv_1 = nn.Sequential(
            nn.Conv2d(44, 66, kernel_size=(1, 3), stride=(1, 1)),
            nn.BatchNorm2d(66),
            nn.LeakyReLU(),
            nn.MaxPool2d((1, 2))
        )
        self.Conv_2 = nn.Sequential(
            nn.Conv2d(66, 66, kernel_size=(1, 3), stride=(1, 1)),
            nn.BatchNorm2d(66),
            nn.LeakyReLU(),
            nn.MaxPool2d((1, 2)),
        )
        self.Conv_3 = nn.Sequential(
            nn.Conv2d(66, 66, kernel_size=(1, 3), stride=(1, 1)),  # 원랜 커널 3,3
            nn.BatchNorm2d(66),
            nn.LeakyReLU(),
            nn.MaxPool2d((1, 2)),
        )
        self.classify = nn.Sequential(
            nn.Linear(in_features=4422, out_features=2)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        ch_n = x.shape[1]

        x = x.view(batch_size * ch_n, 1, -1)
        x = self.morlet_conv(x)
        x = x.view(batch_size, ch_n, filter_n, -1)  # 36,22,filter_n,time_n

        x = self.Conv_0(x)
        x = self.Conv_1(x)
        x = self.Conv_2(x)
        x = self.Conv_3(x)
        x = x.view(x.shape[0], -1)
        global out_linear
        out_linear = x
        x = self.classify(x)

        return F.log_softmax(x, dim=1)
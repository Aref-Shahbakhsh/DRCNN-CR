import math
import torch
import torch.nn as nn
import torch.nn.functional as Fd
import numpy as np
import pandas as p
from torch.utils.data import Dataset, TensorDataset
from torchvision import transforms
import collections
from torch.autograd import Variable
from torch.nn import DataParallel
from math import sqrt
from torch import Tensor
from torch.nn import Linear, Module, Parameter
from typing import Optional, Tuple, Union
from collections import OrderedDict
import pandas as pd
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
class OutfitDataset(Dataset):

    def __init__(self, methyl,label, device=None,transform=None):
        self.methyl = methyl
        self.transform = transform
        self.label = label
        self.device = device
    def __len__(self):
        return len(self.label)
    def __getitem__(self,idx):
        methyl_data = self.methyl[idx]
        label_data = self.label[idx]

        if self.transform:
            methyl_data = torch.from_numpy(np.array(methyl_data))
            label_data = torch.from_numpy(np.array(label_data))

        return (methyl_data), \
               (label_data)

class DownsampleUnit(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, dropout: float):
        super(DownsampleUnit, self).__init__()
        self.norm_act = nn.Sequential(OrderedDict([
            ("0_normalization", nn.BatchNorm1d(in_channels)),
            ("1_activation", nn.GELU()),
        ]))
        self.block = nn.Sequential(OrderedDict([
            ("0_convolution", nn.Conv1d(in_channels, out_channels, 5, stride=stride, padding=2, bias=False)),
            ("1_normalization", nn.BatchNorm1d(out_channels)),
            ("2_activation", nn.GELU()),
            ("3_dropout", nn.Dropout(dropout, inplace=True)),
            ("4_convolution", nn.Conv1d(out_channels, out_channels, 5, stride=1, padding=2, bias=False)),
        ]))
        self.downsample = nn.Conv1d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)

    def forward(self, x):
        x = self.norm_act(x)
        return self.block(x) + self.downsample(x)
class Block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, dropout: float):
        super(Block, self).__init__()
        self.block = nn.Sequential(
            DownsampleUnit(in_channels, out_channels, stride, dropout)
        )

    def forward(self, x):
        return self.block(x)
class DRCNN(nn.Module):
    def __init__(self, width_factor: int, drop: float, in_channels: int, labels: int):
        super(DRCNN, self).__init__()

        self.filters = [16, 1 * 16 * width_factor, 2 * 16 * width_factor, 4 * 16 * width_factor,6 * 16 * width_factor]


        self.conv = nn.Conv1d(in_channels, self.filters[0], 8, stride=3, padding=1, bias=False)
        self.block1 = Block(self.filters[0], self.filters[1], 3, drop)
        self.block2 = Block(self.filters[1], self.filters[2], 3, drop)
        self.block3 = Block(self.filters[2], self.filters[3], 3, drop)
        self.block4 = Block(self.filters[3], self.filters[4], 3, drop)
        self.normalization4 = nn.BatchNorm1d(self.filters[4])
        self.activation5 = nn.GELU()
        self.avgpool6 = nn.AvgPool1d(kernel_size = 5)
        self.flattening7 = nn.Flatten()

        self.res_conv1 = nn.Conv1d(self.filters[0],self.filters[2],kernel_size= 7 ,stride= 9,bias = False,padding = 1)
        self.bn_res_conv1 = nn.BatchNorm1d(num_features=self.filters[2], momentum=0.99, eps=1e-3)
        self.gelu = nn.GELU()

        self.res_conv2 = nn.Conv1d(self.filters[2],self.filters[4],kernel_size= 7 ,stride= 9,bias = False,padding = 2)
        self.bn_res_conv2 = nn.BatchNorm1d(num_features=self.filters[4], momentum=0.99, eps=1e-3)


        self.fcc = nn.Linear(1920, 200)
        self.fcc_drop = nn.Dropout(0.5)
        self.fcc_norm = nn.LayerNorm(200)
        self.fcc_act = nn.GELU()


        self.classification8 = nn.Linear(200, out_features=labels)


    def forward(self, x):

        output = self.conv(x)

        res = self.res_conv1(output)
        res = self.bn_res_conv1(res)
        res = self.gelu(res)


        output = self.block1(output)
        output = self.block2(output)

        output = output + res

        res = self.res_conv2(output)
        res = self.bn_res_conv2(res)
        res = self.gelu(res)

        output = self.block3(output)
        output = self.block4(output)

        output = output + res


        output = self.normalization4(output)
        output = self.activation5(output)

        output = self.avgpool6(output)
        output = self.flattening7(output)


        output = self.fcc(output)
        output = self.fcc_drop(output)
        output = self.fcc_norm(output)
        output = self.fcc_act(output)


        output = self.classification8(output)
        output = nn.functional.log_softmax(output,dim = 1)
        #output = torch.sigmoid(output)
        return output
class StepLR:
    def __init__(self, optimizer, learning_rate: float, total_epochs: int):
        self.optimizer = optimizer
        self.total_epochs = total_epochs
        self.base = learning_rate

    def __call__(self, epoch):
        if epoch < self.total_epochs * 5/10:
            lr = self.base
        elif epoch < self.total_epochs * 7/10:
            lr = self.base * 0.2
        elif epoch < self.total_epochs * 9/10:
            lr = self.base * 0.2 ** 2
        else:
            lr = self.base * 0.2 ** 3

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]
def weight_init(m):
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.Conv1d):
        n = m.kernel_size[0] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
        n = m.num_features
        y = 1.0/np.sqrt(n)
        m.weight.data.uniform_(-y,y)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        n = m.in_features
        y = 1.0/np.sqrt(n)
        m.weight.data.uniform_(-y,y)
        if m.bias is not None:
            m.bias.data.zero_()

"""
Pytorch implementation of MultiAttention 1D CNN (MA1DCNN)

Understanding and Learning Discriminant
Features based on Multiattention 1DCNN for
Wheelset Bearing Fault Diagnosis, Wang et al.

https://ieeexplore.ieee.org/document/8911240
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv1dSamePadding(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(Conv1dSamePadding, self).__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.width = in_channels

        self.padding = self.calculate_padding()
        self.conv_layer = nn.Conv1d(in_channels, out_channels, kernel_size,
                                    stride=stride)

    def calculate_padding(self):
        """
        W/S = (W-K+TP)/S+1    # new W bothers with stride

        # solve for TP (total padding)
        W/S-1 = (W-K+TP)/S
        S(W/S-1) = W-K+TP
        TP = S(W/S-1)-W+K

        TP = W-S-W+K
        TP = K-S
        """
        # p = (self.kernel_size // 2 - 1) * self.stride + 1
        # p = (self.stride * (self.width / self.stride - 1) - self.width + self.kernel_size) / 2
        total_padding = max(0, self.kernel_size - self.stride)
        p1 = total_padding // 2
        p2 = total_padding - p1
        return p1, p2

    def forward(self, x):
        x = F.pad(x, self.padding)
        return self.conv_layer(x)

class CAM(nn.Module):
    def __init__(self, num_filters):
        super(CAM, self).__init__()
        self.num_filters = num_filters
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.conv1 = nn.Conv1d(self.num_filters, self.num_filters // 2, 1, padding="same")
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv1d(self.num_filters // 2, self.num_filters, 1, padding="same")
        self.batchnorm = nn.BatchNorm1d(self.num_filters)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b1 = self.avgpool(x)
        b1 = self.conv1(b1)
        b1 = self.relu(b1)

        b1 = self.conv2(b1)
        b1 = self.batchnorm(b1)
        b1 = self.sigmoid(b1)

        b2 = torch.multiply(x, b1)
        out = x + b2

        return out

class EAM(nn.Module):
    def __init__(self, num_filters, kernel_size):
        super(EAM, self).__init__()
        self.num_filters = num_filters
        self.kernel_size = kernel_size

        self.conv1 = nn.Conv1d(num_filters, 1, 1)
        self.batchnorm = nn.BatchNorm1d(1)
        self.sigmoid = nn.Sigmoid()

        self.conv2 = nn.Conv1d(num_filters, num_filters, kernel_size, padding="same")
        self.relu = nn.ReLU()

    def forward(self, x):
        b1 = self.conv1(x)
        b1 = self.batchnorm(b1)
        b1 = self.sigmoid(b1)

        b2 = self.conv2(x)
        b2 = self.relu(b2)
        b3 = torch.multiply(b1, b2)
        o = x + b3

        return o

class MA1DCNN(nn.Module):
    def __init__(self, num_classes, in_channels):
        super(MA1DCNN, self).__init__()
        self.num_classes = num_classes

        self.conv1 = nn.Conv1d(in_channels, 32, 32, padding="same")
        self.relu1 = nn.ReLU()
        self.eam1 = EAM(32, 32)
        self.cam1 = CAM(32)

        self.conv2 = Conv1dSamePadding(32, 32, 16, stride=2)
        self.relu2 = nn.ReLU()
        self.eam2 = EAM(32, 16)
        self.cam2 = CAM(32)

        self.conv3 = Conv1dSamePadding(32, 64, 9, stride=2)
        self.relu3 = nn.ReLU()
        self.eam3 = EAM(64, 9)
        self.cam3 = CAM(64)

        self.conv4 = Conv1dSamePadding(64, 64, 6, stride=2)
        self.relu4 = nn.ReLU()
        self.eam4 = EAM(64, 6)
        self.cam4 = CAM(64)

        self.conv5 = Conv1dSamePadding(64, 128, 3, stride=4)
        self.relu5 = nn.ReLU()
        self.eam5 = EAM(128, 3)
        self.cam5 = CAM(128)

        self.conv6 = Conv1dSamePadding(128, 128, 3, stride=4)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.linear = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.eam1(x)
        x = self.cam1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.eam2(x)
        x = self.cam2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.eam3(x)
        x = self.cam3(x)

        x = self.conv4(x)
        x = self.relu4(x)
        x = self.eam4(x)
        x = self.cam4(x)

        x = self.conv5(x)
        x = self.relu5(x)
        x = self.eam5(x)
        x = self.cam5(x)

        x = self.conv6(x)
        # x = torch.permute(x, (0, 2, 1))
        x = self.avgpool(x)
        x = torch.squeeze(x)
        x = self.linear(x)

        return F.log_softmax(x, dim=1)

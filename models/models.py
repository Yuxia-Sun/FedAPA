import torch
from torch import nn


class Net(nn.Module):
    def __init__(self, in_channels, num_classes, dim):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=6,
                      kernel_size=5,
                      padding=0,
                      stride=1,
                      bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=6,
                      out_channels=16,
                      kernel_size=5,
                      padding=0,
                      stride=1,
                      bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(dim, 120),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU(inplace=True)
        )
        self.fc = nn.Linear(84, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = torch.flatten(out, start_dim=1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc(out)
        return out


class BaseHeadMerge(nn.Module):
    def __init__(self, base, head):
        super().__init__()
        self.base = base
        self.head = head

    def forward(self, x):
        out = self.base(x)
        out = self.head(out)
        return out

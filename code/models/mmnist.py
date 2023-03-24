import torch.nn as nn
import torch.nn.functional as F


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        # convolution layer
        # i --> input channels
        # 6 --> output channels
        # 5 --> kernel size
        self.conv = nn.Sequential()
        self.conv.add_module("conv1", nn.Conv2d(1, 6, 5))
        self.conv.add_module("relu1", nn.ReLU())
        self.conv.add_module("pool1", nn.MaxPool2d(2))
        self.conv.add_module("conv2", nn.Conv2d(6, 16, 5))
        self.conv.add_module("relu2", nn.ReLU())
        self.conv.add_module("pool2", nn.MaxPool2d(2))

        # full connection layer
        # 16 * 4 * 4 --> input vector dimensions
        # 120 --> output vector dimensions
        self.dense = nn.Sequential()
        self.dense.add_module("dense1", nn.Linear(16 * 4 * 4, 120))
        self.dense.add_module("relu3", nn.ReLU())
        self.dense.add_module("dense2", nn.Linear(120, 84))
        self.dense.add_module("relu4", nn.ReLU())
        self.dense.add_module("dense3", nn.Linear(84, 10))

    def forward(self, x):
        # convolution layer
        conv_out = self.conv(x)

        # x = (n * 16 * 4 * 4) --> n : input channels
        res = conv_out.view(conv_out.size()[0], -1)

        # full connection layer
        out = self.dense(res)

        return out

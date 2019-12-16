## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # input of (1, 224, 224)

        ## Reference paper: https://arxiv.org/pdf/1710.00977.pdf
        self.conv1 = nn.Conv2d(1, 32, 4)  # conv1: (32, 221, 221)
        self.conv2 = nn.Conv2d(32, 64, 3)  # conv2: (64, 53, 53)
        self.conv3 = nn.Conv2d(64, 128, 2)  # conv3: (128, 25, 25)
        self.conv4 = nn.Conv2d(128, 256, 1)  # conv4: (256, 25, 25)

        # maxpooling layers
        self.pool1 = nn.MaxPool2d(4, 4)  # pool1: (32, 55, 55)
        self.pool2 = nn.MaxPool2d(2, 2)  # pool2: (64, 26, 26)
        # pool3: (128, 12, 12)
        # pool4: (256, 6, 6)

        # dropout probablity increase from 0.1 to 0.6, step size = 0.1
        self.drop1 = nn.Dropout(p=0.1)
        self.drop2 = nn.Dropout(p=0.2)
        self.drop3 = nn.Dropout(p=0.3)
        self.drop4 = nn.Dropout(p=0.4)
        self.drop5 = nn.Dropout(p=0.5)
        self.drop6 = nn.Dropout(p=0.6)

        # fully connected layers
        self.fc1 = nn.Linear(256 * 6 * 6, 1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, 136)  # final output of 68 * 2 keypoints

        # dense layers
        self.d1 = nn.Linear(1000, 1000)

    def forward(self, x):

        # 4 conv/elu + pool layers

        x = self.conv1(x)  # 1st convolution
        x = F.relu(x)  # 1st relu activation
        x = self.pool1(x)  # 1st pool
        x = self.drop1(x)  # 1st drop

        x = self.pool2(F.relu(self.conv2(x)))  # 2nd conv/relu + pool
        x = self.drop2(x)  # 2nd drop

        x = self.pool2(F.relu(self.conv3(x)))  # 3rd conv/relu + pool
        x = self.drop3(x)  # 3rd drop

        x = self.pool2(F.relu(self.conv4(x)))  # 4th conv/relu + pool
        x = self.drop4(x)  # 4th drop

        # flatten to prepare for linear layer
        x = x.view(x.size(0), -1)

        # three linear layers with drop out in between
        x = F.relu(self.fc1(x))  # 1st dense layer/elu
        x = self.drop5(x)  # 5th drop

        x = F.relu(self.fc2(x))  # 2nd dense layer/relu
        x = self.drop6(x)  # 6th drop

        x = self.fc3(x)  # 3rd dense layer

        # a modified x, having gone through all the layers of your model, should be returned
        return x

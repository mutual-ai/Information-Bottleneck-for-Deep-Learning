import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
import torch.optim as optim
import torchvision.datasets as dset

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
        # Convolution + average pooling
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        self.avgpool1 = nn.AvgPool2d(kernel_size=2)
     
        # Convolution + max pooling
        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        
        self.dropout = nn.Dropout(p=0.5)

        self.fc1 = nn.Linear(1600, 100)
        self.fc2 = nn.Linear(100, 10)

    
    def forward(self, x):
        # Convolution + average pooling
        out = self.cnn1(x)
        out = self.relu1(out)
        out = self.avgpool1(out)
        
        # Convolution + max pooling
        out = self.cnn2(out)
        out = self.relu2(out)
        out = self.maxpool2(out)
        
        # resize
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        
        # full connect layers
        out = self.fc1(out)
        out = self.fc2(out)
        
        return out
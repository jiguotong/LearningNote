import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
class Net(nn.Module):
    def __init__(self,nclasses=2) -> None:
        super(Net,self).__init__()
        self.nclasses = nclasses

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3,stride = 1,padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2,stride=1)
        self.conv2 = nn.Conv2d(6,12,3)
        self.pool2 = nn.MaxPool2d(2,2)
        self.conv3 = nn.Conv2d(12,self.nclasses,3)
        self.pool3 = nn.MaxPool2d(2,3)

    def forward(self,x):
        ori_size = x.shape[-2:]
        x= self.pool1(torch.relu(self.conv1(x)))
        x= self.pool2(torch.relu(self.conv2(x)))
        x= self.pool3(torch.relu(self.conv3(x)))

        output = F.interpolate(x, ori_size, mode='bilinear', align_corners=True)
        return output

if __name__=='__main__':
    net = Net()
    input = torch.randn(1,3,32,32)
    output = net(input)
    print(input.shape)
    print(output.shape)

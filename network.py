
import os
import torch
from torch import nn
from torchsummaryX import summary

class DeepNetWork(nn.Module):
    def __init__(self,):
        super(DeepNetWork,self).__init__()

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        self.fc = nn.Sequential(
            nn.Linear(4*4*128,256),
            nn.ReLU()
        )

        self.value = nn.Linear(256,1)
        self.advantage = nn.Linear(256,2)

    def forward(self, x):

        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        # 全连接之前展平
        x = x.view(x.size(0),-1)
        x = self.fc(x)

        #Dueling Net

        value = self.value(x)
        advantage = self.advantage(x)

        out = value + advantage - advantage.mean(dim = - 1, keepdim = True)
        # out = advantage
        return out

if __name__ == '__main__':

    net = DeepNetWork()
    summary(net, torch.randn((32, 4, 64, 64)))
    torch.save(net.state_dict(), os.path.join('./', 'net.pth'))


"""
environment encoding model for forst environment
"""
import argparse
import os
import torch
from torch import nn
from torch.autograd import Variable

class Encoder(nn.Module):
    # ref: https://github.com/lxxue/voxnet-pytorch/blob/master/models/voxnet.py
    # adapted from SingleView 2
    def __init__(self, input_size=32, output_size=32):
            super(Encoder, self).__init__()
            input_size = [input_size, input_size]
            self.encoder = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=32, kernel_size=[8,8], stride=[4,4]),  # ~50
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2),                                                   # 25
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=[4,4], stride=[2,2]), # ~11
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2),                                                  # ~5
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=[3,3], stride=[1,1]), # ~4
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2),                                                  # ~2

        )
            x = self.encoder(torch.autograd.Variable(torch.rand([1, 3] + input_size)))
            first_fc_in_features = 1
            for n in x.size()[1:]:
                first_fc_in_features *= n
            print('length of the output of one encoder')
            print(first_fc_in_features)
            self.head = nn.Sequential(
                nn.Linear(first_fc_in_features, 128),
                nn.PReLU(),
                nn.Linear(128, output_size)
            )
            
    def forward(self, x):
        # x shape: BxCxWxH
        #size = x.size()
        #x1 = x.permute(0, 1, 4, 2, 3).reshape(size[0], -1, size[2], size[3])# transpose to Bx(CxD)xWxH

        #x1, x2, x3 = self.encoder1(x1),self.encoder2(x2),self.encoder3(x3)
        #x1, x2, x3 = x1.view(x1.size(0), -1), x2.view(x2.size(0), -1), x3.view(x3.size(0), -1)
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        # cat x1 x2 x3 into x
        #x = torch.cat([x1, x2, x3], dim=1)
        x = self.head(x)
        return x
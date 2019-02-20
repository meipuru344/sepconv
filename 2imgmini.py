
# -*- coding: utf-8 -*-

import getopt
import math
import numpy as np
import os
import PIL
import PIL.Image
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import time
import glob
import random
import cv2
import tensorboardX as tbx
import sys




sys.path.insert(0, './sepconv'); import sepconv


class Network(torch.nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        def Basic(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False)
        )
        # end
        def Subnet():
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=64, out_channels=51, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                torch.nn.Conv2d(in_channels=51, out_channels=51, kernel_size=3, stride=1, padding=1)
            )
            # end
        self.moduleConv1 = Basic(6, 32)
        self.modulePool1 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.moduleConv2 = Basic(32, 64)
        self.modulePool2 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.moduleConv3 = Basic(64, 128)
        self.modulePool3 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.moduleConv4 = Basic(128, 256)
        self.modulePool4 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.moduleConv5 = Basic(256, 512)
        self.modulePool5 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.moduleDeconv5 = Basic(512, 512)
        self.moduleUpsample5 = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.moduleDeconv4 = Basic(512, 256)
        self.moduleUpsample4 = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.moduleDeconv3 = Basic(256, 128)
        self.moduleUpsample3 = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.moduleDeconv2 = Basic(128, 64)
        self.moduleUpsample2 = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.moduleVertical1 = Subnet()
        self.moduleVertical2 = Subnet()
        self.moduleHorizontal1 = Subnet()
        self.moduleHorizontal2 = Subnet()

        self.modulePad = torch.nn.ReplicationPad2d([ int(math.floor(51 / 2.0)), int(math.floor(51 / 2.0)), int(math.floor(51 / 2.0)), int(math.floor(51 / 2.0)) ])

    def forward(self, tensorFirst, tensorSecond):
        tensorJoin = torch.cat([tensorFirst, tensorSecond], 1)
        tensorConv1 = self.moduleConv1(tensorJoin)
        tensorPool1 = self.modulePool1(tensorConv1)

        tensorConv2 = self.moduleConv2(tensorPool1)
        tensorPool2 = self.modulePool2(tensorConv2)

        tensorConv3 = self.moduleConv3(tensorPool2)
        tensorPool3 = self.modulePool3(tensorConv3)

        tensorConv4 = self.moduleConv4(tensorPool3)
        tensorPool4 = self.modulePool4(tensorConv4)

        tensorConv5 = self.moduleConv5(tensorPool4)
        tensorPool5 = self.modulePool5(tensorConv5)

        tensorDeconv5 = self.moduleDeconv5(tensorPool5)
        tensorUpsample5 = self.moduleUpsample5(tensorDeconv5)

        tensorCombine = tensorUpsample5 + tensorConv5

        tensorDeconv4 = self.moduleDeconv4(tensorCombine)
        tensorUpsample4 = self.moduleUpsample4(tensorDeconv4)

        tensorCombine = tensorUpsample4 + tensorConv4

        tensorDeconv3 = self.moduleDeconv3(tensorCombine)
        tensorUpsample3 = self.moduleUpsample3(tensorDeconv3)

        tensorCombine = tensorUpsample3 + tensorConv3

        tensorDeconv2 = self.moduleDeconv2(tensorCombine)
        tensorUpsample2 = self.moduleUpsample2(tensorDeconv2)

        tensorCombine = tensorConv2 + tensorUpsample2

        vertical1 = self.moduleVertical1(tensorCombine)
        horizontal1 = self.moduleHorizontal1(tensorCombine)
        vertical2 = self.moduleVertical2(tensorCombine)
        horizontal2 = self.moduleHorizontal2(tensorCombine)

        return torch.cat((vertical1,horizontal1,vertical2,horizontal2), 3)

#network define
print('Network build')
moduleNetwork = Network()
moduleNetwork = moduleNetwork.cuda()
print('Network built')

if True:
    train = 100
    optimizer = optim.Adamax(moduleNetwork.parameters(), lr=0.001, betas=(0.9, 0.999), eps=0.01, weight_decay=0)
    loss_fn = nn.L1Loss(size_average=True)
    if False:#load pretrained models
        print("Loading Pretrained Model")
        predir = "pretrained_model"
        moduleNetwork.load_state_dict(torch.load("../models/"+predir))

    for epoch in range(100):
        for n in range(train):
            #making train data
            image3b = torch.ones((1,3,178,178)).cuda()##25 pixel wider picture for each direction to synthesis kernel and I1, I,2
            image1b = torch.ones((1,3,178,178)).cuda()
            image1 = torch.ones(1,3,128,128).cuda()
            image2 = torch.ones(1,3,128,128).cuda()
            image3 = torch.ones(1,3,128,128).cuda()

            #forward caluclation
            Kernel = moduleNetwork.forward(image1, image3)
            kernelDiv = torch.chunk(Kernel, 4, dim=3)
            tensorDot1 = sepconv.FunctionSepconv().forward(image1b, kernelDiv[0], kernelDiv[1]).detach()
            tensorDot2 = sepconv.FunctionSepconv().forward(image3b, kernelDiv[2], kernelDiv[3]).detach()
            tensorDot1.requires_grad = True ; tensorDot2.requires_grad = True
            tensorCombine = tensorDot1 + tensorDot2

            #backward caluclation
            loss = loss_fn(tensorCombine, image2)
            value_loss = loss.item()
            loss.backward()
            kgrad1 = sepconv.FunctionSepconv().backward(tensorDot1.grad, (tensorCombine, image1b, kernelDiv[0], kernelDiv[1]))
            kgrad2 = sepconv.FunctionSepconv().backward(tensorDot2.grad, (tensorCombine, image3b, kernelDiv[2], kernelDiv[3]))
            kernelGrad = torch.cat((kgrad1[0],kgrad1[1],kgrad2[0],kgrad2[1]), 3)
            torch.autograd.backward([Kernel], [kernelGrad])

            #using optimizer
            optimizer.step()
            optimizer.zero_grad()

            print("train epoch "+str(epoch)+" n " +str(n)+ " value_loss "+str(value_loss))

    #######Save Models#######
if False:
    torch.save(moduleNetwork.state_dict(), "../models/"+predir)

print('process finished')


# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 21:13:19 2018

@author: zhuoyi wang
"""
import torch
import torch.nn as nn
from torch.autograd import Variable

class CenterLoss(nn.Module):
    """Center loss.
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes=10, feat_dim=2, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu
        
        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))
    
    def forward(self, x, labels):
        """
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (num_classes).
        """
        batch_size = x.size(0)
        dist_matrix = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        dist_matrix.addmm_(1, -2, x, self.centers.t())
        
        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        classes = Variable(classes)
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))
        
        distance = []
        for i in range(batch_size):
            value = dist_matrix[i][mask[i]]
            value = value.clamp(min=1e-12, max=1e+12) # for numerical stability
            distance.append(value)
        distance = torch.cat(distance)
        loss = distance.mean()
        
        return loss
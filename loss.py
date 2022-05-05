#!usr/bin/env python
# -*- coding:utf-8-*-

import torch
import torch.nn as nn
import numpy as np


class classification_loss(nn.Module):
    def __init__(self, weights=None):
        super(classification_loss, self).__init__()
        self.weights = weights
        if self.weights == None:
            self.loss_fun = nn.CrossEntropyLoss()
        else:
            self.loss_fun = [nn.CrossEntropyLoss(torch.Tensor(weight).cuda(), reduction='mean') for weight in weights]

    def forward(self, predict_label, true_label):
        true_label = true_label.long()
        labels_num = true_label.shape[-1]
        loss = 0
        for i in range(labels_num):
            validId = np.where((true_label[:, i].cpu().numpy() == 0) | (true_label[:, i].cpu().numpy() == 1))[0]
            if len(validId) == 0:
                continue
            y_pred = predict_label[:, i * 2:(i + 1) * 2][torch.tensor(validId)]
            y_label = true_label[:, i][torch.tensor(validId)]
            if self.weights == None:
                loss += self.loss_fun(y_pred, y_label.long())
            else:
                loss += self.loss_fun[i](y_pred, y_label.long())
        return loss

class regression_loss(nn.Module):
    def __init__(self, ):
        super(regression_loss, self).__init__()
        self.loss_fun = nn.MSELoss()

    def forward(self, predict_label, true_label):
        true_label = true_label.float().cuda()
        labels_num = true_label.shape[-1]
        loss = 0
        for i in range(labels_num):
            loss += self.loss_fun(predict_label[:, i], true_label[:, i])
        return loss


class regression_ratio_loss(nn.Module):
    def __init__(self, ratio_list):
        super(regression_ratio_loss, self).__init__()
        self.loss_fun = nn.MSELoss()
        self.ratio_list = ratio_list

    def forward(self, predict_label, true_label):
        true_label = true_label.float().cuda()
        labels_num = true_label.shape[-1]
        loss = 0
        for i in range(labels_num):
            loss += self.loss_fun(predict_label[:, i], true_label[:, i]) * self.ratio_list[i]**2
        return loss

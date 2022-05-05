import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import *
import numpy as np
from sklearn import metrics
from sklearn.metrics import precision_recall_curve
from copy import deepcopy
from utils import *
from confidence import *

class MetaLearner(nn.Module):
    def __init__(self, model, loss_function, correct_function):
        super(MetaLearner, self).__init__()
        self.update_step = 5  ## task-level inner update steps
        self.update_step_test = 5
        self.net = model
        self.meta_lr = 0.0005
        self.base_lr = 0.0005
        print("meta_lr:", self.meta_lr,"      base_lr:  ", self.base_lr)
        self.meta_optim = torch.optim.Adam(self.net.parameters(), lr=self.meta_lr)
        # self.scheduler = lr_scheduler.StepLR(self.meta_optim, step_size=10, gamma=0.8)
        # self.meta_optim = torch.optim.SGD(self.net.parameters(),
        #                         lr=self.meta_lr, weight_decay=1e-5, momentum = 0.9)

        self.loss_function = loss_function
        self.correct_function = correct_function


    def forward(self, x_spt, y_spt, x_qry, y_qry, tasks_type):
        # 初始化
        task_num = len(x_spt)
        loss_list_qry = [0 for _ in range(self.update_step + 1)]
        correct_list = [0 for _ in range(self.update_step + 1)]

        y_spt = y_spt.squeeze(dim=0).cuda()
        y_qry = y_qry.squeeze(dim=0).cuda()
        for i in range(task_num):
            if "classification" in tasks_type:
                validId = np.where((y_spt[i].view(-1).cpu().numpy() == 0) | (y_spt[i].view(-1).cpu().numpy() == 1))[0]
                if len(validId) == 0:
                    continue

            ## 第0步更新
            y_hat = self.net(x_spt[i], params=list(self.net.parameters()))  # (ways * shots, ways)
            loss = self.loss_function(y_hat, y_spt[i])

            grad = torch.autograd.grad(loss, self.net.parameters())

            tuples = zip(grad, self.net.parameters())  ## 将梯度和参数\theta一一对应起来
            # fast_weights这一步相当于求了一个\theta - \alpha*\nabla(L)
            fast_weights = list(map(lambda p: p[1] - self.base_lr * p[0], tuples))
            # 在query集上测试，计算准确率
            # 这一步使用更新前的数据
            with torch.no_grad():
                y_hat = self.net(x_qry[i], list(self.net.parameters()))
                loss_qry = self.loss_function(y_hat, y_qry[i])
                loss_list_qry[0] += loss_qry
                correct = self.correct_function(y_hat, y_qry[i])
                correct_list[0] += correct

            # 使用更新后的数据在query集上测试。
            with torch.no_grad():
                # aa2 = list(self.net.parameters())[0]
                # aa3 = list(fast_weights)[0]
                y_hat = self.net(x_qry[i], fast_weights)
                loss_qry = self.loss_function(y_hat, y_qry[i])
                loss_list_qry[1] += loss_qry
                correct = self.correct_function(y_hat, y_qry[i])
                correct_list[1] += correct

            for k in range(1, self.update_step):
                y_hat = self.net(x_spt[i], params=fast_weights)
                loss = self.loss_function(y_hat, y_spt[i])
                grad = torch.autograd.grad(loss, fast_weights)
                tuples = zip(grad, fast_weights)
                fast_weights = list(map(lambda p: p[1] - self.base_lr * p[0], tuples))

                if k < self.update_step - 1:
                    with torch.no_grad():
                        y_hat = self.net(x_qry[i], params=fast_weights)
                        loss_qry = self.loss_function(y_hat, y_qry[i])
                        loss_list_qry[k + 1] += loss_qry
                else:
                    y_hat = self.net(x_qry[i], params=fast_weights)
                    loss_qry = self.loss_function(y_hat, y_qry[i])
                    loss_list_qry[k + 1] += loss_qry

                with torch.no_grad():
                    correct = self.correct_function(y_hat, y_qry[i])
                    correct_list[k + 1] += correct


        loss_qry = loss_list_qry[-1] / task_num
        self.meta_optim.zero_grad()  # 梯度清零
        loss_qry.backward()
        self.meta_optim.step()
        # print("lr   ", self.meta_optim.state_dict()['param_groups'][0]['lr'])
        accs = np.array(correct_list) / task_num
        loss = np.array(loss_list_qry) / task_num
        return accs, loss

    def finetunning(self,task_index,  x_spt, y_spt, test_load):
        y_spt = y_spt.long().cuda()
        test_task_pred_list = []
        test_task_label_list = []
        test_task_smi_list = []

        new_net = deepcopy(self.net)
        y_hat = new_net(x_spt, list(new_net.parameters()))
        loss = self.loss_function(y_hat, y_spt)
        grad = torch.autograd.grad(loss, new_net.parameters())
        fast_weights = list(map(lambda p: p[1] - self.base_lr * p[0], zip(grad, new_net.parameters())))

        # if task_index == 0:
        #     # confidence(new_net, fast_weights)
        #     confidence_cls(new_net, fast_weights)

        # 在query集上测试，计算准确率
        # 这一步使用更新前的数据
        with torch.no_grad():
            y_pred_list = []
            y_pred_list2 = []
            y_labels_list = []
            y_smi_list = []
            for query_x, query_y in test_load:
                y_hat = new_net(query_x, list(new_net.parameters()))
                y_pred_list.extend(y_hat)

                # 使用更新后的数据在query集上测试。
                y_hat2 = new_net(query_x, fast_weights)

                y_pred_list2.extend(y_hat2)
                y_labels_list.extend(query_y)
                y_smi_list.extend(query_x)

            y_pred = torch.stack(y_pred_list, dim=0)
            y_labels = torch.stack(y_labels_list, dim=0)
            # y_smi = torch.stack(y_smi_list, dim=0)

            test_task_pred_list.append(y_pred)
            test_task_label_list.append(y_labels)
            test_task_smi_list.append(y_smi_list)

            y_pred2 = torch.stack(y_pred_list2, dim=0)
            test_task_pred_list.append(y_pred2)
            test_task_label_list.append(y_labels)
            test_task_smi_list.append(y_smi_list)


        for k in range(1, self.update_step_test):
            y_hat = new_net(x_spt, params=fast_weights)
            loss = self.loss_function(y_hat, y_spt)
            grad = torch.autograd.grad(loss, fast_weights)
            fast_weights = list(map(lambda p: p[1] - self.base_lr * p[0], zip(grad, fast_weights)))

            # if task_index == 0:
            #     # confidence(new_net, fast_weights)
            #     confidence_cls(new_net, fast_weights)

            with torch.no_grad():
                y_pred_list = []
                y_labels_list = []
                y_smi_list = []
                for query_x, query_y in test_load:
                    y_hat = new_net(query_x, fast_weights)
                    y_pred_list.extend(y_hat)
                    y_labels_list.extend(query_y)
                    y_smi_list.extend(query_x)

                y_pred = torch.stack(y_pred_list, dim=0)
                y_labels = torch.stack(y_labels_list, dim=0)
                # y_smi = torch.stack(y_smi_list, dim=0)

                test_task_pred_list.append(y_pred)
                test_task_label_list.append(y_labels)
                test_task_smi_list.append(y_smi_list)

        del new_net
        return test_task_pred_list, test_task_label_list, test_task_smi_list
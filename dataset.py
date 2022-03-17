import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from utils import *
import random
import joblib
import numpy as np
import pandas as pd



class MyDataset_train(data.Dataset):
    def __init__(self, data_dir, dataset_name, k_spt, k_query, tasks, batch_tasks_num, batchsz, mean_list=None, std_list=None):
        super(MyDataset_train, self).__init__()
        self.dataset_name = dataset_name
        self.data_dir = data_dir

        self.mean_list = mean_list
        self.std_list = std_list

        self.k_spt_num = k_spt

        self.k_query = k_query  # for evaluation
        self.tasks = tasks
        self.batch_tasks_num = batch_tasks_num
        self.batchsz = batchsz

        self.all_smi, self.all_labels = self.load_excel()

        self.batchs_data = []

        self.create_batch_muti_cls()

    def load_excel(self):
        all_smi = {}
        all_labels = {}
        for file in self.tasks:
            path = os.path.join(self.data_dir, file)
            scaffold = file.split(".")[0]
            smiles_task = pd.read_csv(path)
            all_smi[scaffold] = smiles_task.smiles.values

            labels = np.array(smiles_task.iloc[:, data_dict[self.dataset_name]["start"]:data_dict[self.dataset_name]["end"]])
            if "qm9".__eq__(self.dataset_name):
                all_labels[scaffold] = qm9_normalized(self.mean_list, self.std_list, labels)
            else:
                all_labels[scaffold] = labels

        return all_smi, all_labels


    def create_batch_muti_cls(self):
        """
        create batch for meta-learning.
        ×episode× here means batch, and it means how many sets we want to retain.
        :param episodes: batch size
        :return:
        """
        for b in range(self.batchsz):  # for each batch
            # 1.select n_way classes randomly
            # selected_cls = random.choices(self.tasks, k=self.task_num)  #duplicate
            selected_cls = random.sample(self.all_smi.keys(), k=self.batch_tasks_num)  # no duplicate
            np.random.shuffle(selected_cls)
            support_x = []
            support_y = []
            query_x = []
            query_y = []
            for cls in selected_cls:
                # 2. select k_shot + k_query for each class
                sample_num = self.all_smi[cls].shape[0]

                idx = np.random.choice(sample_num, self.k_spt_num, False)
                np.random.shuffle(idx)
                cls_support_x = list(self.all_smi[cls][idx])
                cls_support_y = list(self.all_labels[cls][idx])


                all_index = [i for i in range(sample_num) if i not in idx]
                query_idx = np.random.choice(all_index, self.k_query, False)
                np.random.shuffle(query_idx)

                cls_query_x = list(self.all_smi[cls][query_idx])
                cls_query_y = list(self.all_labels[cls][query_idx])

                support_x.append(cls_support_x)
                support_y.append(cls_support_y)
                query_x.append(cls_query_x)
                query_y.append(cls_query_y)

            # support_x = np.array(support_x)
            support_y = np.array(support_y)

            # query_x = np.array(query_x)
            query_y = np.array(query_y)

            self.batchs_data.append([support_x, support_y, query_x, query_y])

    def __getitem__(self, item):
        x_spt, y_spt, x_qry, y_qry = self.batchs_data[item]
        return x_spt, y_spt, x_qry, y_qry

    def __len__(self):
        return len(self.batchs_data)


class MyDataset_test(data.Dataset):
    def __init__(self, data_dir, dataset_name, k_spt, task, mean_list=None, std_list=None):
        super(MyDataset_test, self).__init__()
        self.dataset_name = dataset_name
        self.data_dir = data_dir

        self.mean_list = mean_list
        self.std_list = std_list

        self.k_spt_num = k_spt

        self.task = task
        self.scaffold = self.task.split(".")[0]

        self.all_smi, self.all_labels = self.load_excel()

        self.support_x, self.support_y = self.select_spt()

        # delete support
        self.idx = [i for i in range(len(self.all_smi)) if self.all_smi[i] not in self.support_x]
        self.smiles = self.all_smi[self.idx]
        self.labels = self.all_labels[self.idx]

    def load_excel(self):
        path = os.path.join(self.data_dir, self.task)
        smiles_task = pd.read_csv(path)
        all_smi = smiles_task.smiles.values

        labels = np.array(smiles_task.iloc[:, data_dict[self.dataset_name]["start"]:data_dict[self.dataset_name]["end"]])
        if "qm9".__eq__(self.dataset_name):
            all_labels = qm9_normalized(self.mean_list, self.std_list, labels)
        else:
            all_labels = labels

        return all_smi, all_labels

    def select_spt(self):

        sample_num = len(self.all_smi)
        idx = np.random.choice(sample_num, self.k_spt_num, False)
        np.random.shuffle(idx)
        support_x = self.all_smi[idx]
        support_y = self.all_labels[idx]

        return support_x, np.array(support_y)

    def __getitem__(self, item):
        return self.smiles[item], self.labels[item]

    def __len__(self):
        return len(self.smiles)
#!/usr/bin/env python3
import os
import sys
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, InMemoryDataset, Dataset
import torch_geometric.transforms as T
from rdkit import Chem, RDConfig
from rdkit.Chem import ChemicalFeatures, MolFromSmiles, AllChem
from rdkit import Chem
from rdkit.Chem import Draw
from sklearn import metrics
from sklearn.metrics import precision_recall_curve, mean_squared_error
import random


data_dict = {
    # muti_classification
    "pcba":{"start": 0,"end": 128},
    "tox21":{"start": 0,"end": 12},
    "toxcast":{"start": 1,"end": 618},
    "muv":{"start": 0,"end": 17},

    #  sing_classification
    "hiv":{"start": 2,"end": 3},
    "jnk3":{"start": 1,"end": 2},
    "gsk3":{"start": 1,"end": 2},

    #  muti_regression
    "qm9": {"start": 5, "end": 17},
    "chembl": {"start": 2, "end": 4},

    #  sing_regression
    "ld50": {"start": 1, "end": 2},
    "zinc": {"start": 0, "end": 1},
    "pdbbind_full": {"start": 1, "end": 2},
     }

def predictive_entropy(predictions):
    epsilon = sys.float_info.min

    predictive_entropy = - np.mean(predictions, axis=0) * np.log(np.mean(predictions, axis=0) + epsilon)
    # predictive_entropy = -np.sum(np.mean(predictions, axis=0) * np.log(np.mean(predictions, axis=0) + epsilon),
    #                              axis=0)

    return predictive_entropy


def classification_score(predict_label, true_label):
    true_label = true_label.long()
    tasks_num = true_label.shape[-1]
    y_pred_tasks = []
    y_true_tasks = []
    for i in range(tasks_num):
        validId = np.where((true_label[:, i].cpu().numpy() == 0) | (true_label[:, i].cpu().numpy() == 1))[0]
        if len(validId) == 0:
            continue
        pred_qry = F.softmax(predict_label[:, i * 2:(i + 1) * 2][torch.tensor(validId)].detach().cpu(), dim=-1)
        y_pred_tasks.extend(pred_qry[:, 1].view(-1).numpy())
        y_true_tasks.extend(true_label[:, i][torch.tensor(validId)].cpu().numpy())

    trn_roc = metrics.roc_auc_score(y_true_tasks, y_pred_tasks)
    trn_prc = metrics.auc(precision_recall_curve(y_true_tasks, y_pred_tasks)[1],
                           precision_recall_curve(y_true_tasks, y_pred_tasks)[0])
    return trn_roc, trn_prc


def classification_correct(predict_label, true_label):
    true_label = true_label.long()
    labels_num = true_label.shape[-1]
    correct = 0
    total = 0
    for i in range(labels_num):

        validId = np.where((true_label[:, i].cpu().numpy() == 0) | (true_label[:, i].cpu().numpy() == 1))[0]
        if len(validId) == 0:
            continue
        pred_qry = F.softmax(predict_label[:, i * 2:(i + 1) * 2][torch.tensor(validId)], dim=-1).argmax(dim=-1)
        correct += torch.eq(pred_qry, true_label[:, i][torch.tensor(validId)].cuda()).sum().item()
        total += len(validId)
    return correct / total


def regression_rmse_score(predict_label, true_label):
    true_label = true_label.float().cpu().numpy()
    predict_label = predict_label.cpu().numpy()
    labels_num = true_label.shape[-1]
    loss = 0
    for i in range(labels_num):
        loss += np.sqrt(mean_squared_error(predict_label[:, i], true_label[:, i]))
    return loss

def regression_muti_rmse_score(predict_label, true_label):
    true_label = true_label.float().cpu().numpy()
    predict_label = predict_label.cpu().numpy()
    labels_num = true_label.shape[-1]
    loss = []
    for i in range(labels_num):
        loss.append(np.sqrt(mean_squared_error(predict_label[:, i], true_label[:, i])))
    return loss

def regression_mae_mean_score(predict_label, true_label):
    true_label = true_label.float().cpu().numpy()
    predict_label = predict_label.cpu().numpy()
    labels_num = true_label.shape[-1]
    mae = []
    for i in range(labels_num):
        mae.append(mean_squared_error(predict_label[:, i], true_label[:, i]))
    return np.mean(mae)


def regression_mae_score(predict_label, true_label):
    true_label = true_label.float().cpu().numpy()
    predict_label = predict_label.cpu().numpy()
    labels_num = true_label.shape[-1]
    mae = []
    for i in range(labels_num):
        mae.append(mean_squared_error(predict_label[:, i], true_label[:, i]))

    return mae

def qm9_mae_score(predict_label, true_label, std_list):
    true_label = true_label.float().cpu().numpy()
    predict_label = predict_label.cpu().numpy()

    tasks = [
        "mu", "alpha", "homo", "lumo", "gap", "r2", "zpve", "u0", "u298", "h298", "g298", "cv"
    ]
    eval_MAE_list = {}
    y_val_list = {}
    y_pred_list = {}
    for i, task in enumerate(tasks):
        y_pred_list[task] = np.array([])
        y_val_list[task] = np.array([])
        eval_MAE_list[task] = np.array([])

    for i, task in enumerate(tasks):
        mae = mean_squared_error(predict_label[:, i], true_label[:, i])

        y_pred_list[task] = np.concatenate([y_pred_list[task], predict_label[:, i]])
        y_val_list[task] = np.concatenate([y_val_list[task], true_label[:, i]])
        eval_MAE_list[task] = np.concatenate([eval_MAE_list[task], [mae]])

    eval_MAE_normalized = np.array([eval_MAE_list[task].mean() for i, task in enumerate(tasks)])
    eval_MAE = np.multiply(eval_MAE_normalized, np.array(std_list))

    return eval_MAE



def get_qm9_mean_std(data_dir):
    tasks = [
        "mu", "alpha", "homo", "lumo", "gap", "r2", "zpve", "u0", "u298", "h298", "g298", "cv"
    ]
    all_files = os.listdir(data_dir)
    file_csv = []
    for file in all_files:
        path = os.path.join(data_dir, file)
        file_csv.append(pd.read_csv(path))
    all_data = pd.concat(file_csv)

    mean_list = []
    std_list = []
    mad_list = []
    ratio_list = []
    for task in tasks:
        mean = all_data[task].mean()
        mean_list.append(mean)
        std = all_data[task].std()
        std_list.append(std)
        mad = all_data[task].mad()
        mad_list.append(mad)
        ratio_list.append(std / mad)

    return mean_list, std_list, mad_list, ratio_list


def qm9_normalized(mean_list, std_list, data):
    labels_num = data.shape[-1]
    for i in range(labels_num):
        data[:,i] = (data[:,i] - mean_list[i])/std_list[i]
    return data

def get_weights(data_root, dataset_name, labels_num):
    data_dir = os.path.join(data_root, dataset_name)
    all_files = os.listdir(data_dir)
    file_csv = []
    for file in all_files:
        path = os.path.join(data_dir, file)
        file_csv.append(pd.read_csv(path))
    all_data = pd.concat(file_csv)

    labels = np.array(all_data.iloc[:, data_dict[dataset_name]["start"]:data_dict[dataset_name]["end"]])

    weights = []
    for i in range(labels_num):
        # print(i)
        negative_num = len(np.where(labels[:,i] == 0)[0])
        positive_num = len(np.where(labels[:,i] == 1)[0])
        if negative_num == 0:
            negative_num = 1
        if positive_num == 0:
            positive_num = 1
        weights.append([(negative_num + positive_num) / negative_num, \
                        (negative_num + positive_num) / positive_num])
    print(weights)
    return weights

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from copy import deepcopy, copy
import torch.utils.data as data
from MetaLearner import *
from model import GAT
from smiles_feature import *
from loss import *
from dataset import *
import random
import pandas as pd

device = torch.device('cuda')

seed = 199
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

### 准备数据迭代器
k_spt = 10 # support data 的个数
batch_tasks_num = 8   # batch中 task 的个数
k_query = 16  # query data 的个数


p_dropout = 0.2
fingerprint_dim = 200
radius = 2
T = 2
batchsz = 1000

test_batchsz = 32
epochs = 10

print("batchsz:", batchsz, ",    seed :", seed)

num_atom_features = 39
num_bond_features = 10

# dataset_name, tasks_type, train_tasks_num, labels_num = "pcba", "muti_classification", 25, 128
# dataset_name, tasks_type, train_tasks_num, labels_num = "tox21", "muti_classification", 8, 12
# dataset_name, tasks_type, train_tasks_num, labels_num = "toxcast", "muti_classification", 8, 617
# dataset_name, tasks_type, train_tasks_num, labels_num = "muv", "muti_classification", 122, 17

# dataset_name, tasks_type, train_tasks_num, labels_num = "hiv", "sing_classification", 54, 1
# dataset_name, tasks_type, train_tasks_num, labels_num = "jnk3", "sing_classification", 50, 1
# dataset_name, tasks_type, train_tasks_num, labels_num = "gsk3", "sing_classification", 28, 1

dataset_name, tasks_type, train_tasks_num, labels_num = "pdbbind_full", "sing_regression", 8, 1
# dataset_name, tasks_type, train_tasks_num, labels_num = "zinc", "sing_regression", 10, 1
# dataset_name, tasks_type, train_tasks_num, labels_num = "ld50", "sing_regression", 8, 1

# dataset_name, tasks_type, train_tasks_num, labels_num = "qm9", "muti_regression", 35, 12
# dataset_name, tasks_type, train_tasks_num, labels_num = "chembl", "muti_regression", 159, 2

data_root = "./data/"
data_dir = os.path.join(data_root, dataset_name)
all_files = os.listdir(data_dir)
csv_files = [file for file in all_files if "csv".__eq__(file.split(".")[-1])]
train_tasks = random.sample(csv_files, train_tasks_num)
test_tasks = [task for task in csv_files if task not in train_tasks]
print(test_tasks)

batch_test_tasks_num = len(test_tasks)
print("dataset name:", dataset_name, ",    test task num :", batch_test_tasks_num, "     k_spt:", k_spt)

####################
row1 = []
smiles_task = []
for file in train_tasks:
    path = os.path.join(data_dir, file)
    csv_data = pd.read_csv(path)
    row1 = list(csv_data.columns.values)
    smiles_task.append(pd.read_csv(path))
a = pd.concat(smiles_task)

train_num_int = round(len(a) * 0.8)


df = pd.DataFrame(a[:train_num_int], columns=row1)
df.to_csv("./data2/"+dataset_name+"_train.csv", encoding="GBK", index=None)


df = pd.DataFrame(a[train_num_int:], columns=row1)
df.to_csv("./data2/"+dataset_name+"_val.csv", encoding="GBK", index=None)

smiles_task = []
for file in test_tasks:
    path = os.path.join(data_dir, file)
    smiles_task.append(pd.read_csv(path))
a = pd.concat(smiles_task)


df = pd.DataFrame(a, columns=row1)
df.to_csv("./data2/"+dataset_name+"_test.csv", encoding="GBK", index=None)


mean_list = None
std_list = None

if "qm9".__eq__(dataset_name):
    mean_list, std_list, mad_list, ratio_list = get_qm9_mean_std(data_dir)


data_train = MyDataset_train(data_dir, dataset_name, k_spt, k_query, train_tasks, batch_tasks_num, batchsz, mean_list, std_list)
dataset_train = data.DataLoader(dataset=data_train, batch_size=1, shuffle=True)

# muti  test
data_test = [MyDataset_test(data_dir, dataset_name, k_spt, task, mean_list, std_list) for task in test_tasks]
dataset_test = [data.DataLoader(dataset=data_i, batch_size=test_batchsz, shuffle=True) for data_i in data_test]



feature_filename = os.path.join(data_root, dataset_name+".pkl")
if os.path.isfile(feature_filename):
    feature_dicts = joblib.load(open(feature_filename, "rb"))
else:
    feature_dicts = save_smiles_dicts_from_dir(data_dir, feature_filename)

if "sing_regression".__eq__(tasks_type):

    loss_function = regression_loss()
    correct_function = regression_rmse_score
    test_score = regression_rmse_score
    output_units_num = labels_num

elif "muti_regression".__eq__(tasks_type):

    loss_function = regression_loss()
    correct_function = regression_rmse_score
    test_score = regression_muti_rmse_score
    output_units_num = labels_num

    if "qm9".__eq__(dataset_name):
        loss_function = regression_ratio_loss(ratio_list)
        correct_function = regression_mae_mean_score
        test_score = qm9_mae_score
elif "classification" in tasks_type:
    weights = get_weights(data_root, dataset_name, labels_num)
    loss_function = classification_loss(weights=weights)
    correct_function = classification_correct
    test_score = classification_score
    output_units_num = labels_num * 2
else:
    print("tasks_type  error")

model = GAT(radius, T, num_atom_features, num_bond_features,
                    fingerprint_dim, output_units_num, p_dropout, feature_dicts).to(device)

meta = MetaLearner(model, loss_function, correct_function).to(device)


for epoch in range(epochs):

    for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(dataset_train):
        # print("epoch:", epoch, "step:", step)

        accs_train, loss = meta(x_spt, y_spt, x_qry, y_qry, tasks_type)
        if step % 10 == 0:
            print("epoch:", epoch, "step:", step)
            print(accs_train)
            # print(loss)

        if step % 50 == 0:
            pred_tasks = [[] for _ in test_tasks]
            label_tasks = [[] for _ in test_tasks]
            for task_index, (test_tmp, test_load) in enumerate(zip(data_test, dataset_test)):
                # print(task_index)
                support_x, support_y = test_tmp.support_x, test_tmp.support_y

                pred_list, label_list = meta.finetunning(support_x, torch.tensor(support_y), test_load)
                pred_tasks[task_index] = pred_list
                label_tasks[task_index] = label_list
            # accs_res = np.array(accs).mean(axis=1).astype(np.float16)
            pred_all = [[] for _ in range(len(pred_tasks[0]))]
            label_all = [[] for _ in range(len(label_tasks[0]))]
            for i in range(len(test_tasks)):
                print('\n 测试集 第 ' + str(i) + ' 个任务:')
                for t, (pred, label) in enumerate(zip(pred_tasks[i], label_tasks[i])):
                    print(correct_function(pred, label),"\t", end='')
                    pred_all[t].extend(pred)
                    label_all[t].extend(label)
            print('\n 测试集 score:')
            if "qm9".__eq__(dataset_name):
                for t, (pred, label) in enumerate(zip(pred_all, label_all)):
                    print(qm9_mae_score(torch.stack(pred, dim=0), torch.stack(label, dim=0), std_list),"\n", end='')
                print('\n')
            else:
                for t, (pred, label) in enumerate(zip(pred_all, label_all)):
                    print(test_score(torch.stack(pred, dim=0), torch.stack(label, dim=0)),"\n", end='')
                print('\n')

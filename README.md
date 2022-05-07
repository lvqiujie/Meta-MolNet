# Meta-MolNet: A cross domain benchmark for few examples drug discovery

Meta-MolNet is the first standard benchmark platform for measuring model generalization and uncertainty quantification capabilities. Meta-MolNet manages a molecular benchmark collection to measure the effectiveness of the proposed algorithm in uncertainty quantification and generalization evaluation, which contains broad public datasets of high molecules/scaffolds split by scaffolds. 

![Image text](https://github.com/lol88/Meta-MolNet/blob/master/image.png)

By publishing AI-ready data, evaluation frameworks, and baseline results, we aim to encourage researchers to focus on the new and challenging problem of achieving reliable domain generalization with few examples data. We hope to see the Meta-MolNet suite become a comprehensive resource for the AI-assisted drug discovery community.




## Tutorials
We provide tutorials to get started with Meta-MolNet:

### Data

The data folder contains 12 benchmark datasets. Each dataset has been divided according to the scaffold, and molecules belonging to the same scaffold are in a csv file named after the scaffold.

### Data Loaders

```
dataset_name, tasks_type, train_tasks_num, labels_num = "hiv", "sing_classification", 54, 1
data_train = MyDataset_train(data_dir, dataset_name, k_spt, k_query, train_tasks, batch_tasks_num, batchsz, mean_list, std_list)
```

### Train

```
python Meta-MolNet.py
```

## Cite

```
@article{Lv2022Meta-MolNet,
  title={Meta-MolNet: A cross domain benchmark for few examples drug discovery},
  author={Qiujie Lv, Guanxing Chen, Ziduo Yang, Weihe Zhong and Calvin Yu-Chian Chen},
  journal={},
  year={2022}
}
```











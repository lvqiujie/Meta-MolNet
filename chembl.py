from rdkit import Chem
import numpy as np
import openpyxl
import pandas as pd
import os
import json
from scaffold_split import *
from smiles_feature import *
import joblib

scaffold_list = []
all_scaffolds_dict = {}


df = pd.read_csv(os.path.join('./data/raw/chembl.csv'), error_bad_lines=False)
for indexs in df.index:
    # if indexs  == 30000:
    #     joblib.dump(all_scaffolds_dict, './data/chembl_scaffolds_dict.pkl')
    print(indexs)
    line = df.loc[indexs].values[0].split(";")
    line1 = df.loc[indexs].values[0]
    id = line[0]
    smi = line[30][1:-1]
    ALogP = line[8][1:-1]
    LogD = line[19][1:-1]

    if smi is None or len(smi) == 0 or "None".__eq__(smi):
        continue

    if ALogP is None or LogD is None \
            or len(ALogP) == 0 or len(LogD) == 0 \
            or "None".__eq__(ALogP) or "None".__eq__(LogD):
        # print("@@@@@@@@@@ ", ALogP, LogD)
        continue

    try:

        molgraph = graph_from_smiles(smi)
        molgraph.sort_nodes_by_degree('atom')
        arrayrep = array_rep_from_smiles(molgraph)

        scaffold = generate_scaffold(smi)
        if "C1CCNCC1".__eq__(scaffold):
            scaffold = "c1ccncc1"
        elif "C1CCCCC1".__eq__(scaffold):
            scaffold = "c1ccccc1"
    except:
        print(smi)
        continue
    # if "".__eq__(scaffold):
    #     print(smiles)
    scaffold_list.append(scaffold)
    if scaffold not in all_scaffolds_dict:
        all_scaffolds_dict[scaffold] = [[id, smi, ALogP, LogD]]
    else:
        all_scaffolds_dict[scaffold].append([id, smi, ALogP, LogD])

joblib.dump(all_scaffolds_dict, './data/chembl_scaffolds_dict.pkl')

all_scaffolds_dict = joblib.load('./data/chembl_scaffolds_dict.pkl')

def save_excel(name, data):

    df = pd.DataFrame(data, columns=["chemblID", "smiles", "ALogP", "LogD"])
    df.to_csv(name,encoding="GBK", index=None)

num_all = 0
num_10 = 0
num_300 = 0
num_1000 = 0
for k, v in all_scaffolds_dict.items():
    if "".__eq__(k):
        continue
    # print(k, v)
    if len(v) > 1000:
        num_1000 += 1
        num_all += len(v)
        save_excel("./data/chembl/" + k + ".csv", v)
    elif len(v) > 300:
        save_excel("./data/chembl/"+k+".csv", v)
        num_300 += 1
        num_all += len(v)
    elif len(v) > 10:
        num_10 += 1
        # save_excel("./data/chembl/"+k+".xlsx", v)
print(num_10, num_300, num_1000)
print(num_all)

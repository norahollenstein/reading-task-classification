import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from scipy import stats
import  seaborn as sns

result_dir ="../results/"
dataset= "zuco2"
embeddings = "glove"

if dataset == "zuco2":
    subj_start = "Y"
if dataset == "zuco1":
    subj_start = "Z"

feature = "text_only"

colnames=["random_seed","test_acc", "avg_precision", "avg_recall", "avg_fscore"]

print(dataset, feature)
for filename in os.listdir(result_dir):
    if filename.endswith("-"+embeddings+".txt"):
        #print(filename)
        subj = filename.replace("_saccTrue-"+embeddings+".txt", "").replace("_saccFalse-"+embeddings+".txt", "")[-3:]
        print(subj)
        #if feature in filename and subj.startswith(subj_start):
        print(filename)
        infile = pd.read_csv(result_dir + filename, sep=" ", header=None, comment="l",usecols=[8,11,13,15,17], names=colnames)
        print(infile)
        infile['subject'] = subj
        print(infile)

print(np.mean(infile['test_acc']), np.std(infile['test_acc']))

glove_baseline = 0.6570135951042175
random_baseline = 0.5

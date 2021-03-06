import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from scipy import stats
import  seaborn as sns

result_dir ="../results/"
dataset= "zuco2"
feature = "eye_tracking"
feature_add = "saccTrue"
task = "read-task"

if dataset == "zuco1":
    subj_start = "Z"
    glove_baseline = 0.5375586797793707
    bert_baseline = 0.578 #0.5799686948458354
    flesch_baseline = 0.57 #0.5774647887323944
    color = "#FF9700"
if dataset == "zuco2":
    subj_start = "Y"
    glove_baseline = 0.6570135951042175
    bert_baseline = 0.6533936858177185
    flesch_baseline = 0.5308108108108107
    color = "#C00000"

random_baseline = 0.5

colnames=["random_seed","test_acc", "avg_precision", "avg_recall", "avg_fscore"]

print(dataset, feature)
all_results_pd = pd.DataFrame(columns=colnames)
no_files = 0
for filename in os.listdir(result_dir):
    if filename.endswith(".txt") and feature in filename and feature_add in filename and task in filename:
        subj = filename.replace("_saccTrue.txt", "").replace("_saccFalse.txt", "")[-3:]
        #print(subj)
        if subj.startswith(subj_start):
            print(filename)
            infile = pd.read_csv(result_dir + filename, sep=" ", header=None, comment="l", usecols=[7,10,12,14,16], names=colnames)
            no_files += 1
            if len(infile) != 5:
                print(len(infile), "lines in file!!")
            infile['subject'] = subj
            all_results_pd = pd.concat([all_results_pd, infile])

print(no_files, " files")
all_results_pd = all_results_pd.drop_duplicates()
results = all_results_pd.sort_values(by=['test_acc'])

subjects = ['YAC', 'YAG', 'YAK', 'YDG', 'YDR', 'YFR', 'YFS', 'YHS', 'YIS', 'YLS', 'YMD', 'YRK', 'YRP', 'YSD', 'YSL', 'YTL', "ZAB", "ZDM", "ZDN", "ZGW", "ZJM", "ZJN", "ZKB", "ZKH", "ZKW","ZMG", "ZPH"]

order = []
for s in results.subject.unique():
    subj_results = results.loc[results['subject'] == s]
    order.append((s, np.mean(subj_results['test_acc'])))
    print(s, np.mean(subj_results['test_acc']))

order_sorted = sorted(order, key=lambda x: x[1])
order_sorted = [f[0] for f in order_sorted]

print("Median accuracy:", np.median(results['test_acc']))

ax = sns.pointplot(x="subject", y="test_acc", data=results, ci="sd", color=color, s=80, order=order_sorted, join=False)
median = np.median(results['test_acc'])
mad = np.median(np.absolute(results['test_acc'] - np.median(results['test_acc'])))
ax.axhline(median, ls='--', color="grey", label="median")
plt.text(-0.49, median + 0.01, "{:.2f}".format(median), color="grey", fontweight='bold')
ax.axhspan(median + mad, median - mad, alpha=0.3, color='grey', label="MAD")
ax.axhline(random_baseline, ls='-', color="#4e4f52", label="random baseline")
ax.axhline(bert_baseline, ls=':', color="darkblue", label="word embedding baseline")
ax.axhline(flesch_baseline, ls='-.', color="black", label="text difficulty baseline")
plt.ylim(0.45, 1)
plt.title(feature + " " + feature_add, fontsize=14)
plt.ylabel("accuracy")
handles, labels = ax.get_legend_handles_labels()
# sort both labels and handles by labels
labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
plt.legend(handles, labels)
plt.savefig("plots/wordLevel_" + feature + "_"+ feature_add + "_" + dataset + ".pdf")
plt.show()
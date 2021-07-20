import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from scipy import stats
import  seaborn as sns

result_dir ="../results/"
dataset= "zuco1"

if dataset == "zuco2":
    subj_start = "Y"
    glove_baseline = 0.6570135951042175
    bert_baseline = 0.6533936858177185
if dataset == "zuco1":
    subj_start = "Z"
    glove_baseline = 0.5375586797793707
    bert_baseline = 0.5799686948458354

random_baseline = 0.5

feature = "eye_tracking"
feature_add = "saccFalse"
task = "tasks-cross-subj"

colnames=["random_seed","test_acc", "avg_precision", "avg_recall", "avg_fscore"]

print(dataset, feature)
all_results_pd = pd.DataFrame(columns=colnames)
for filename in os.listdir(result_dir):
    if filename.endswith(".txt") and feature in filename and feature_add in filename and task in filename:
        print(filename)
        subj = filename.replace("_saccTrue.txt", "").replace("_saccFalse.txt", "")[-3:]
        print(subj)
        if subj.startswith(subj_start):
            print(filename)
            infile = pd.read_csv(result_dir + filename, sep=" ", header=None, comment="l", usecols=[7,10,12,14,16], names=colnames)
            infile['subject'] = subj
            all_results_pd = pd.concat([all_results_pd, infile])

all_results_pd = all_results_pd.drop_duplicates()
results = all_results_pd.sort_values(by=['test_acc'])

subjects = ['YAC', 'YAG', 'YAK', 'YDG', 'YDR', 'YFR', 'YFS', 'YHS', 'YIS', 'YLS', 'YMD', 'YRK', 'YRP', 'YSD', 'YSL', 'YTL', "ZAB", "ZDM", "ZDN", "ZGW", "ZJM", "ZJN", "ZKB", "ZKH", "ZKW","ZMG", "ZPH"]
colors = sns.color_palette("flare", len(subjects))
colors_by_subject = [colors[subjects.index(s)] for s in results.subject.unique()]

order = []
for s in results.subject.unique():
    subj_results = results.loc[results['subject'] == s]
    order.append((s, np.mean(subj_results['test_acc'])))
    print(s, np.mean(subj_results['test_acc']))

order_sorted = sorted(order, key=lambda x: x[1])
order_sorted = [f[0] for f in order_sorted]

print("Median accuracy:", np.median(results['test_acc']))

ax = sns.pointplot(x="subject", y="test_acc", data=results, ci="sd", palette=colors_by_subject, s=80, order=order_sorted)
median = np.median(results['test_acc'])
mad = np.median(np.absolute(results['test_acc'] - np.median(results['test_acc'])))
ax.axhline(median, ls='--', color="grey", label="median")
plt.text(-0.49, median + 0.01, "{:.2f}".format(median), color="grey", fontweight='bold')
ax.axhspan(median + mad, median - mad, alpha=0.3, color='grey', label="MAD")
ax.axhline(random_baseline, ls='-.', color="darkblue", label="random baseline")
ax.axhline(bert_baseline, ls=':', color="darkblue", label="text baseline")
plt.ylim(0.4, 1)
plt.title(feature + " " + feature_add)
plt.ylabel("accuracy")
plt.legend()
plt.savefig("plots/wordLevel_" + task + "_" + feature + "_"+ feature_add + "_" + dataset + ".pdf")
plt.show()
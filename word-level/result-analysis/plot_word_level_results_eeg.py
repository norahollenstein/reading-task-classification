import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from scipy import stats

result_dir ="./new_oct15/lstm/"
dataset= "zuco1"

if dataset == "zuco2":
    subj_start = "Y"
if dataset == "zuco1":
    subj_start = "Z"

features = ["eeg_theta", "eeg_beta", "eeg_gamma","eeg_alpha"]

colnames=["subj", "lstm_dim", "lstm_layers", "dense_dim", "dropout", "batch_size", "epochs", "lr", "embedding_type",
          "random_seed", "train_acc", "val_acc", "test_acc", "test_std", "avg_precision", "std_precision",
          "avg_recall", "std_recall", "avg_fscore", "std_fscore", "threshold", "folds", "training_time", 'best_ep', 'patience', 'min_delta', "model"]



for f in features:
    print(f)
    all_results_pd = pd.DataFrame(columns=colnames)
    for filename in os.listdir(result_dir):
        if filename.endswith(".txt"):
            #print(filename)
            subj = filename.replace("_saccTrue.txt", "").replace("_saccFalse.txt", "")[-3:]
            print(subj)
            if f in filename and subj.startswith(subj_start):
                print(filename)
                infile = pd.read_csv(result_dir + filename, sep=" ", header=None, comment="l", names=colnames)
                #print(infile)
                infile['subj'] = subj
                all_results_pd = pd.concat([all_results_pd, infile])

    #print(all_results_pd.head())
    #print(len(all_results_pd))

    subjs =[]
    accs = []
    stds = []
    for i, subj in enumerate(all_results_pd['subj'].unique()):
        accuracies = all_results_pd.loc[all_results_pd['subj'] == subj, 'test_acc'].values
        if len(accuracies) == 10:
            avg_acc = np.mean(accuracies)
            std_err_acc = np.std(accuracies)
            print(subj, avg_acc)
            subjs.append(subj)
            accs.append(avg_acc)
            stds.append(std_err_acc)
        else:
            print("incorrect number of random seeds:", f, len(accuracies), subj)
            sys.exit("!!!")

    median_subj_acc = np.median(accs)
    mad = stats.median_abs_deviation(accs)
    print(median_subj_acc, mad)
    chance=0.5
    zipped_lists = zip(accs, stds, subjs)
    sorted_pairs = sorted(zipped_lists, reverse=True)
    print(sorted_pairs)


    fig, ax = plt.subplots(figsize=(0.25*len(subjs), 4))
    cmaplist = ["#44A2C4", "#337F9A", "#10997D", "#66BB97", "#92D050", "#D5E600", "#FFEB00", "#FFB14C", "#DC7810", "#C00000", "#A30071", "#A072C4", "#642D8F", "#203864", "#2E75B6", "#53BAFF"]
    x_pos = np.arange(len(sorted_pairs))

    for i, s in enumerate(sorted_pairs):
        ax.plot(i, s[0], marker='o', color=cmaplist[list(all_results_pd['subj'].unique()).index(s[2])], markersize=8)
        ax.errorbar(i, s[0], yerr=s[1], color=cmaplist[list(all_results_pd['subj'].unique()).index(s[2])], alpha=0.8, capsize=3)

    #ax.set_ylim(bottom=ymin, top=ymax)
    #ax.set_ylim(ax.get_ylim()[::-1])
    ax.set_ylabel('Accuracy', fontsize=14)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([x[2] for x in sorted_pairs], fontsize=11, rotation=90)
    plt.axhline(y=median_subj_acc, color='black', linestyle='--', label='median')
    ax.axhspan(median_subj_acc-mad, median_subj_acc+mad, alpha=0.1, facecolor='black', edgecolor=None, label="MAD")
    plt.axhline(y=chance, color='grey', linestyle='--', label='chance')
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=True)
    #ax.set_xticklabels(list(file['model'].unique()), fontsize=8, rotation=45)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #ax.set_title("Word level ET features Sacc False")
    plt.draw()
    plt.title(f)
    plt.savefig("wordLevel_"+f+"_"+dataset+".png")
    plt.show()

import config
from datetime import date
import h5py
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def prepare_output_files():
    """Create all output files required for reporting results"""

    if config.level == "word":
        result_file = open("results/" + str(
            date.today()) + "_svm_results_" + config.class_task + "_" + config.dataset + "_random" + str(
            config.randomized_labels) + "_" + config.kernel + "_FixFeats.csv", "a")
        all_runs_result_file = open("results/" + str(
            date.today()) + "_svm_all_runs_" + config.class_task + "_" + config.dataset + "_random" + str(
            config.randomized_labels) + "_" + config.kernel + "_FixFeats.csv", "a")
        coef_file = open("results/" + str(
            date.today()) + "_svm_coefficients_" + config.class_task + "_" + config.dataset + "_random" + str(
            config.randomized_labels) + "_" + config.kernel + "_FixFeats.csv", "a")
    else:
        result_file = open("results/" + str(
            date.today()) + "_svm_results_" + config.class_task + "_" + config.dataset + "_random" + str(
            config.randomized_labels) + "_" + config.kernel + ".csv", "a")
        all_runs_result_file = open("results/" + str(
            date.today()) + "_svm_all_runs_" + config.class_task + "_" + config.dataset + "_random" + str(
            config.randomized_labels) + "_" + config.kernel + ".csv", "a")
        coef_file = open("results/" + str(
            date.today()) + "_svm_coefficients_" + config.class_task + "_" + config.dataset + "_random" + str(
            config.randomized_labels) + "_" + config.kernel + ".csv", "a")


    return result_file, all_runs_result_file, coef_file


def read_mat_file(filename):
    """Read MATLAB files with EEG data"""
    mat_file = h5py.File(filename, 'r')
    sentence_data = mat_file['sentenceData']

    return sentence_data


def plot_feature_distribution(subj, dataset, feature_dict, feature_set):
    """Plot feature distribution for a single feature"""

    colors = ["#44A2C4", "#B3D882"]

    feature_file = open("feature-plots/"+feature_set+"-"+dataset+".csv", "a")

    data = pd.DataFrame(columns=["subject", "feat", "label"])
    for i, (x,y) in enumerate(feature_dict.items()):
        data.loc[i] = [x[:3], y[0], y[1]]


    fig, ax = plt.subplots()
    print(subj, np.mean(data['feat']), np.std(data['feat']), np.min(data['feat']), np.max(data['feat']), file=feature_file)
    ax = sns.violinplot(x="subject", y="feat", hue="label", data=data, palette=colors)#, inner="stick")
    #for axis in fi.axes.flatten():
     #   print("qxis:", axis)
    ax.collections[0].set_edgecolor("#337F9A")  # "#337F9A"
    ax.collections[1].set_edgecolor("#92D050") # "#337F9A"
    #ax.get_children()[5:].set_color("#92D050")


    ax.set_title(feature_set)
    ax.set(xticklabels=[])
    ax.set_xlabel('')
    ax.set_ylabel('')
    fig.savefig("feature-plots/"+ feature_set + "_" +subj+".pdf")
    plt.close()
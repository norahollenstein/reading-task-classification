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

    feature_file = open("feature-plots/"+feature_set+"-"+dataset+".csv", "a")

    data = pd.DataFrame(columns=["subject", "feat", "label"])
    for i, (x,y) in enumerate(feature_dict.items()):
        data.loc[i] = [x[:3], y[0], y[1]]

    fig, ax = plt.subplots()
    print(subj, np.mean(data['feat']), np.std(data['feat']), np.min(data['feat']), np.max(data['feat']), file=feature_file)
    ax = sns.violinplot(x="subject", y="feat", hue="label", data=data, palette="muted")
    ax.set_title(subj + ", mean " + "{:.2f}".format(np.mean(data['feat'])) + ", std " + "{:.2f}".format(np.std(data['feat'])))
    fig.savefig("feature-plots/"+ feature_set + "_" +subj+".pdf")
    plt.close()
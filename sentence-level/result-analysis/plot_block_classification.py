import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def plot_results_detailed(results, dataset):
    features = results.feature_set.unique()
    print(features)
    print(len(results))
    results = results.drop_duplicates()
    print(len(results))

    subjects = ['YAC', 'YAG', 'YAK', 'YDG', 'YDR', 'YFR', 'YFS', 'YHS', 'YIS', 'YLS', 'YMD', 'YRK', 'YRP', 'YSD', 'YSL', 'YTL', "ZAB", "ZDM", "ZDN", "ZGW", "ZJM", "ZJN", "ZKB", "ZKH", "ZKW","ZMG", "ZPH"]

    colors = sns.color_palette("flare", len(subjects))

    ax = sns.barplot(x=results["feature_set"], y=results["accuracy"])
    ax.set_xticklabels(results["feature_set"], rotation=90)
    plt.title("Block classification")
    plt.tight_layout()
    plt.savefig("plots/block_class_"+dataset+".pdf")
    plt.show()


def main():
    result_file = "../results/2021-07-13_svm_results_blocks_zuco2_randomFalse_linear.csv"
    results = pd.read_csv(result_file, delimiter=" ", names=["subject", "feature_set", "accuracy", "std", "features", "samples", "runs"])
    print(results.head())
    dataset = result_file.split("_")[4]
    plot_results_detailed(results, dataset)




if __name__ == '__main__':
    main()

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def check_std_between_runs(results):
    """Get the average standard deviation across all subjects"""
    runs = [5, 10, 50, 100]
    for r in runs:
        rslt_df = results[results['runs'] == r]
        print("number of runs:", np.unique(rslt_df["runs"]))
        print("average std:", np.mean(rslt_df["std"]))


def plot_results_detailed(results, dataset):
    features = results.feature_set.unique()
    print(features)
    print(len(results))
    results = results.drop_duplicates()
    print(len(results))

    subjects = ['YAC', 'YAG', 'YAK', 'YDG', 'YDR', 'YFR', 'YFS', 'YHS', 'YIS', 'YLS', 'YMD', 'YRK', 'YRP', 'YSD', 'YSL', 'YTL', "ZAB", "ZDM", "ZDN", "ZGW", "ZJM", "ZJN", "ZKB", "ZKH", "ZKW","ZMG", "ZPH"]

    colors = sns.color_palette("flare", len(subjects))

    for f in features:
        feature_results = results[results['feature_set'] == f]
        feature_results = feature_results.sort_values(by=['accuracy'])
        colors_by_subject = [colors[subjects.index(s)] for s in feature_results.subject.unique()]

        print("Mean accuracy:", f, np.mean(feature_results['accuracy']))

        ax = sns.pointplot(x="subject", y="accuracy", data=feature_results, ci="sd", palette=colors_by_subject, s=70)
        ax.set_title(f)
        median = np.median(feature_results['accuracy'])
        mad = np.median(np.absolute(feature_results['accuracy'] - np.median(feature_results['accuracy'])))
        ax.axhline(median, ls='--', color="grey", label="median")
        plt.text(-0.49, median+0.01, "{:.2f}".format(median), color="grey", fontweight='bold')
        ax.axhspan(median+mad, median-mad, alpha=0.3, color='grey', label="MAD")
        ax.axhline(0.5, ls='-.', color="darkblue", label="random")
        plt.ylim(0.49,1)
        plt.legend()
        plt.savefig("plots/"+f+"-"+dataset+".pdf")
        plt.show()


def main():
    result_file = "../results/2021-04-14_svm_results_tasks_zuco1_randomFalse_linear.csv"
    results = pd.read_csv(result_file, delimiter=" ", names=["subject", "feature_set", "accuracy", "std", "features", "samples", "runs"])
    #check_std_between_runs(results)

    result_file_all = "../results/2021-04-14_svm_all_runs_tasks_zuco2_randomFalse_linear.csv"
    results = pd.read_csv(result_file_all, delimiter=" ", names=["subject", "feature_set", "accuracy", "samples", "run"])
    dataset = result_file_all.split("_")[5]
    plot_results_detailed(results, dataset)



if __name__ == '__main__':
    main()

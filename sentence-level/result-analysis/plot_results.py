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

    if dataset == "zuco1":
        flesch_baseline = 0.5774647887323944
    if dataset == "zuco2":
        flesch_baseline = 0.5308108108108107
    random_baseline = 0.5

    for f in features:
        feature_results = results[results['feature_set'] == f]
        feature_results = feature_results.sort_values(by=['accuracy'])
        colors_by_subject = [colors[subjects.index(s)] for s in feature_results.subject.unique()]

        print("Mean accuracy:", f, np.mean(feature_results['accuracy']))

        order = []
        for s in feature_results.subject.unique():
            subj_results = feature_results.loc[feature_results['subject'] == s]
            order.append((s, np.mean(subj_results['accuracy'])))

        order_sorted = sorted(order, key=lambda x: x[1])
        order_sorted = [f[0] for f in order_sorted]

        ax = sns.pointplot(x="subject", y="accuracy", data=feature_results, ci="sd", palette=colors_by_subject, s=70, order=order_sorted)
        ax.set_title(f)
        median = np.median(feature_results['accuracy'])
        mad = np.median(np.absolute(feature_results['accuracy'] - np.median(feature_results['accuracy'])))
        ax.axhline(median, ls='--', color="grey", label="median")
        plt.text(-0.49, median+0.01, "{:.2f}".format(median), color="grey", fontweight='bold')
        ax.axhspan(median+mad, median-mad, alpha=0.3, color='grey', label="MAD")
        ax.axhline(random_baseline, ls='-.', color="darkblue", label="random")
        ax.axhline(flesch_baseline, ls='-.', color="darkblue", label="random")
        plt.ylim(0.49,1)
        plt.legend()
        plt.savefig("plots/"+f+"-"+dataset+".pdf")
        plt.show()


def cross_subj_results(results, dataset):
    results = results.sort_values(by=['accuracy'])
    print(results)

    subjects = ['YAC', 'YAG', 'YAK', 'YDG', 'YDR', 'YFR', 'YFS', 'YHS', 'YIS', 'YLS', 'YMD', 'YRK', 'YRP', 'YSD', 'YSL',
                'YTL', "ZAB", "ZDM", "ZDN", "ZGW", "ZJM", "ZJN", "ZKB", "ZKH", "ZKW", "ZMG", "ZPH"]
    colors = sns.color_palette("flare", len(subjects))
    colors_by_subject = [colors[subjects.index(s)] for s in results.subject.unique()]

    ax = sns.pointplot(x="subject", y="accuracy", data=results, ci="sd", palette=colors_by_subject, s=70)
    ax.set_title(results['feature_set'][0])
    median = np.median(results['accuracy'])
    mad = np.median(np.absolute(results['accuracy'] - np.median(results['accuracy'])))
    ax.axhline(median, ls='--', color="grey", label="median")
    plt.text(-0.49, median + 0.01, "{:.2f}".format(median), color="grey", fontweight='bold')
    ax.axhspan(median + mad, median - mad, alpha=0.3, color='grey', label="MAD")
    ax.axhline(0.5, ls='-.', color="darkblue", label="random")
    plt.ylim(0.4, 1)
    plt.legend()
    plt.savefig("plots/cross-subj-" + dataset + ".pdf")
    plt.show()



def main():
    result_file = "../results/2021-04-14_svm_results_tasks_zuco1_randomFalse_linear.csv"
    results = pd.read_csv(result_file, delimiter=" ", names=["subject", "feature_set", "accuracy", "std", "features", "samples", "runs"])
    #check_std_between_runs(results)

    result_file_all = "../results/2021-04-14_svm_all_runs_tasks_zuco2_randomFalse_linear.csv"
    results = pd.read_csv(result_file_all, delimiter=" ", names=["subject", "feature_set", "accuracy", "samples", "run"])
    dataset = result_file_all.split("_")[5]
    plot_results_detailed(results, dataset)

    result_file_cross = "../results/2021-04-15_svm_all_runs_tasks-cross-subj_zuco2_randomFalse_linear.csv"
    results_cross = pd.read_csv(result_file_cross, delimiter=" ", names=["subject", "feature_set", "accuracy", "samples", "run"])
    dataset = result_file_cross.split("_")[5]
    #cross_subj_results(results_cross, dataset)



if __name__ == '__main__':
    main()

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



def plot_results_detailed(results, dataset, task):
    print(results.head())
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
        ax.axhline(random_baseline, ls='-.', color="darkblue", label="random baseline")
        if task != "sessions" or task != "blocks":
            ax.axhline(flesch_baseline, ls=':', color="darkblue", label="Flesch baseline")
        plt.ylim(0.49,1.03)
        plt.legend()
        plt.savefig("plots/"+task+"-"+f+"-"+dataset+".pdf")
        plt.show()


def plot_results_fixations(results, dataset):

    print(len(results))
    results = results.drop_duplicates()
    print(len(results))

    results.loc[results['feature_set'] == "fix_order_raw_eeg_electrodes", 'feature_set'] = 100
    results.loc[results['feature_set'] == "fix_order_raw_eeg_electrodes_75%", 'feature_set'] = 75
    results.loc[results['feature_set'] == "fix_order_raw_eeg_electrodes_50%", 'feature_set'] = 50
    results.loc[results['feature_set'] == "fix_order_raw_eeg_electrodes_20%", 'feature_set'] = 20
    results.loc[results['feature_set'] == "fix_order_raw_eeg_electrodes_10%", 'feature_set'] = 10
    features = results.feature_set.unique()
    print(features)

    for f in features:
        mean_f = results.loc[results['feature_set'] == f, 'accuracy'].mean()
        results = results.append({'subject':"MEAN", 'feature_set': f, 'accuracy':mean_f, 'samples':"-", 'run':"-"}, ignore_index=True)

    print(results.subject.unique())

    subjects = ['YAC', 'YAG', 'YAK', 'YDG', 'YDR', 'YFR', 'YFS', 'YHS', 'YIS', 'YLS', 'YMD', 'YRK', 'YRP', 'YSD', 'YSL', 'YTL', "ZAB", "ZDM", "ZDN", "ZGW", "ZJM", "ZJN", "ZKB", "ZKH", "ZKW","ZMG", "ZPH", "MEAN"]

    colors = sns.color_palette("flare", len(subjects))

    if dataset == "zuco1":
        flesch_baseline = 0.5774647887323944
    if dataset == "zuco2":
        flesch_baseline = 0.5308108108108107
    random_baseline = 0.5

    colors_by_subject = [colors[subjects.index(s)] for s in results.subject.unique()]
    colors_by_subject[-1] = "#000000"

    labels = [None, None, None, 'YDG', 'YDR', 'YFR', 'YFS', 'YHS', 'YIS', 'YLS', 'YMD', 'YRK', 'YRP', 'YSD', 'YSL', 'YTL', "ZAB", "ZDM", "ZDN", "ZGW", "ZJM", "ZJN", "ZKB", "ZKH", "ZKW","ZMG", "ZPH", "MEAN"]

    ax = sns.lineplot(x="feature_set", y="accuracy", data=results, hue="subject", marker='o', ci='sd', palette=colors_by_subject, legend=False)#, s=70)#, order=order_sorted)
    #ax.set_title("Fixation percentage")
    ax.axhline(random_baseline, ls='-.', color="darkblue", label="random baseline")
    ax.axhline(flesch_baseline, ls=':', color="darkblue", label="Flesch baseline")
    for f in features:
        ax.axvline(f, ls='--', color="lightgrey")
    plt.xlim(10,100)
    plt.ylim(0.4,1)
    plt.xlabel("percentage of fixations")
    plt.legend()
    plt.xticks([10,20,50,75,100], [10,20,50,75,100])
    plt.savefig("plots/fixFeats_"+dataset+".pdf")
    plt.show()


def cross_subj_results(results, dataset):
    results = results.sort_values(by=['accuracy'])

    if dataset == "zuco1":
        flesch_baseline = 0.5774647887323944
    if dataset == "zuco2" or "zucoAll":
        flesch_baseline = 0.5308108108108107
    random_baseline = 0.5

    subjects = ['YAC', 'YAG', 'YAK', 'YDG', 'YDR', 'YFR', 'YFS', 'YHS', 'YIS', 'YLS', 'YMD', 'YRK', 'YRP', 'YSD', 'YSL',
                'YTL', "ZAB", "ZDM", "ZDN", "ZGW", "ZJM", "ZJN", "ZKB", "ZKH", "ZKW", "ZMG", "ZPH"]
    colors = sns.color_palette("flare", len(subjects))
    colors_by_subject = [colors[subjects.index(s)] for s in results.subject.unique()]

    print("Mean accuracy:", np.mean(results['accuracy']))

    ax = sns.pointplot(x="subject", y="accuracy", data=results, ci="sd", palette=colors_by_subject, s=80)
    ax.set_title(results['feature_set'][0])
    median = np.median(results['accuracy'])
    mad = np.median(np.absolute(results['accuracy'] - np.median(results['accuracy'])))
    ax.axhline(median, ls='--', color="grey", label="median")
    plt.text(-0.49, median + 0.01, "{:.2f}".format(median), color="grey", fontweight='bold')
    ax.axhspan(median + mad, median - mad, alpha=0.3, color='grey', label="MAD")
    ax.axhline(random_baseline, ls='-.', color="darkblue", label="random baseline")
    ax.axhline(flesch_baseline, ls=':', color="darkblue", label="Flesch baseline")
    plt.ylim(0, 1)
    plt.legend()
    plt.savefig("plots/cross-subj-" + results.feature_set.unique()[0] + "_"+ dataset + ".pdf")
    plt.show()



def main():
    result_file = "../results/2021-04-14_svm_results_tasks_zuco1_randomFalse_linear.csv"
    results = pd.read_csv(result_file, delimiter=" ", names=["subject", "feature_set", "accuracy", "std", "features", "samples", "runs"])
    #check_std_between_runs(results)

    result_file_all = "../results/tasks-zuco2-final.csv"
    results = pd.read_csv(result_file_all, delimiter=" ", names=["subject", "feature_set", "accuracy", "samples", "run"])
    dataset = result_file_all.split("-")[1]
    task = "tasks"
    plot_results_detailed(results, dataset, task)

    # Session classification with ZuCo 1 SR data
    result_file_all = "../results/2021-07-13_svm_all_runs_sessions_zuco1sr_randomFalse_linear.csv"
    results = pd.read_csv(result_file_all, delimiter=" ",
                          names=["subject", "feature_set", "accuracy", "samples", "run"])
    dataset = result_file_all.split("_")[5]
    task = result_file_all.split("_")[4]
    #plot_results_detailed(results, dataset, task)

    # Block classification with ZuCo 2 data
    result_file_all = "../results/2021-07-13_svm_all_runs_blocks_zuco2_randomFalse_linear.csv"
    results = pd.read_csv(result_file_all, delimiter=" ",
                          names=["subject", "feature_set", "accuracy", "samples", "run"])
    dataset = result_file_all.split("_")[5]
    task = result_file_all.split("_")[4]
    #plot_results_detailed(results, dataset, task)

    result_file_cross = "../results/2021-04-19_svm_all_runs_tasks-cross-subj_zucoAll_randomFalse_linear.csv"
    results_cross = pd.read_csv(result_file_cross, delimiter=" ", names=["subject", "feature_set", "accuracy", "samples", "run"])
    dataset = result_file_cross.split("_")[5]
    #cross_subj_results(results_cross, dataset)

    result_file_fix = "../results/2021-04-21_svm_all_runs_tasks_zuco1_randomFalse_linear_FixFeats.csv"
    results_fix = pd.read_csv(result_file_fix, delimiter=" ",
                                names=["subject", "feature_set", "accuracy", "samples", "run"])
    dataset = result_file_fix.split("_")[5]
    print(dataset)
    #plot_results_fixations(results_fix, dataset)



if __name__ == '__main__':
    main()

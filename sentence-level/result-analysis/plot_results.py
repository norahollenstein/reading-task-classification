import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


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

    if dataset == "zuco1":
        bert_baseline = 0.578  # 0.5799686948458354
        flesch_baseline = 0.57  # 0.5774647887323944
        color = "#FF9700"
    if dataset == "zuco2":
        bert_baseline = 0.6533936858177185
        flesch_baseline = 0.5308108108108107
        color = "#C00000"
    random_baseline = 0.5

    for f in features:
        feature_results = results[results['feature_set'] == f]
        feature_results = feature_results.sort_values(by=['accuracy'])

        order = []
        for s in feature_results.subject.unique():
            subj_results = feature_results.loc[feature_results['subject'] == s]
            order.append((s, np.median(subj_results['accuracy'])))

        order_sorted = sorted(order, key=lambda x: x[1])
        order_sorted = [f[0] for f in order_sorted]

        ax = sns.pointplot(x="subject", y="accuracy", data=feature_results, ci="sd", color=color, s=70, order=order_sorted, join=False)
        ax.set_title(f, fontsize=14)
        median = np.median(feature_results['accuracy'])
        mad = np.median(np.absolute(feature_results['accuracy'] - np.median(feature_results['accuracy'])))
        print(f, median, mad)
        ax.axhline(median, ls='--', color="grey", label="median")
        plt.text(-0.49, median+0.01, "{:.2f}".format(median), color="grey", fontweight='bold')
        ax.axhspan(median+mad, median-mad, alpha=0.3, color='grey', label="MAD")
        ax.axhline(random_baseline, ls='-', color="#4e4f52", label="random baseline")
        if task != "sessions" or task != "blocks":
            ax.axhline(bert_baseline, ls=':', color="darkblue", label="word embedding baseline")
            ax.axhline(flesch_baseline, ls='-.', color="black", label="text difficulty baseline")
        plt.ylim(0.49,1.03)
        handles, labels = ax.get_legend_handles_labels()
        # sort both labels and handles by labels
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
        plt.legend(handles, labels)
        plt.savefig("plots/"+task+"-"+f+"-"+dataset+".pdf")
        #plt.show()


def plot_results_fixations(results, dataset):

    print(len(results))
    results = results.drop_duplicates()
    print(len(results))

    # for gamma:
    results.loc[results['feature_set'] == 'fix_electrode_features_gamma', 'feature_set'] = 100
    results.loc[results['feature_set'] == 'fix_electrode_features_gamma_75%', 'feature_set'] = 75
    results.loc[results['feature_set'] == 'fix_electrode_features_gamma_50%', 'feature_set'] = 50
    results.loc[results['feature_set'] == 'fix_electrode_features_gamma_20%', 'feature_set'] = 20
    results.loc[results['feature_set'] == 'fix_electrode_features_gamma_10%', 'feature_set'] = 10

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
    #ax.axhline(random_baseline, ls='-.', color="darkblue", label="random baseline")
    #ax.axhline(flesch_baseline, ls=':', color="darkblue", label="Flesch baseline")
    for f in features:
        ax.axvline(f, ls='--', color="lightgrey")
    plt.xlim(10,100)
    if dataset == "zuco1":
        plt.ylim(0.9,1.01)
    if dataset == "zuco2":
        plt.ylim(0.6, 1.0)
    plt.xlabel("percentage of fixations")
    mean_line = mpatches.Patch(color='black', label='mean')
    #plt.legend(handles=[mean_line])
    plt.xticks([10,20,50,75,100], [10,20,50,75,100])
    plt.savefig("plots/fixFeats_gamma_"+dataset+".pdf")
    plt.show()


def cross_subj_results(results, dataset):
    print(results.head())
    features = results.feature_set.unique()
    print(features)
    print(len(results))
    results = results.drop_duplicates()
    print(len(results))

    if dataset == "zuco1":
        bert_baseline = 0.578  # 0.5799686948458354
        flesch_baseline = 0.57  # 0.5774647887323944
        color = "#FF9700"
    if dataset == "zuco2":
        bert_baseline = 0.6533936858177185
        flesch_baseline = 0.5308108108108107
        color = "#C00000"
    if dataset == "zucoAll":
        # baseline for both??
        bert_baseline = 0.6533936858177185
        flesch_baseline = 0.5308108108108107
        color = "#993DA7"

    random_baseline = 0.5

    for feature in features:

        feature_results = results[results['feature_set'] == feature]
        feature_results = feature_results.sort_values(by=['accuracy'])

        order = []
        for s in feature_results.subject.unique():
            subj_results = feature_results.loc[feature_results['subject'] == s]
            order.append((s, np.median(subj_results['accuracy'])))

        order_sorted = sorted(order, key=lambda x: x[1])
        order_sorted = [f[0] for f in order_sorted]

        ax = sns.pointplot(x="subject", y="accuracy", data=feature_results, ci="sd", color=color, s=80, join=False, order=order_sorted)
        median = np.median(feature_results['accuracy'])
        mad = np.median(np.absolute(feature_results['accuracy'] - np.median(feature_results['accuracy'])))
        print("Mean accuracy:", feature, np.median(feature_results['accuracy']), mad)
        ax.axhline(median, ls='--', color="grey", label="median")
        plt.text(-0.49, median + 0.01, "{:.2f}".format(median), color="grey", fontweight='bold')
        ax.axhspan(median + mad, median - mad, alpha=0.3, color='grey', label="MAD")
        ax.axhline(random_baseline, ls='-', color="#4e4f52", label="random baseline")
        ax.axhline(bert_baseline, ls=':', color="darkblue", label="word embedding baseline")
        ax.axhline(flesch_baseline, ls='-.', color="black", label="text difficulty baseline")
        plt.ylim(0, 1)
        handles, labels = ax.get_legend_handles_labels()
        # sort both labels and handles by labels
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
        plt.legend(handles, labels)
        plt.title(feature, fontsize= 14)
        plt.savefig("plots/cross-subj-" + feature + "_"+ dataset + ".pdf")
        plt.show()



def main():
    result_file = "../results/2021-04-14_svm_results_tasks_zuco1_randomFalse_linear.csv"
    results = pd.read_csv(result_file, delimiter=" ", names=["subject", "feature_set", "accuracy", "std", "features", "samples", "runs"])
    #check_std_between_runs(results)

    # Task classification
    """
    result_file_all = "../results/tasks-zuco2-final.csv"
    results = pd.read_csv(result_file_all, delimiter=" ", names=["subject", "feature_set", "accuracy", "samples", "run"])
    dataset = result_file_all.split("-")[1]
    task = "tasks"
    plot_results_detailed(results, dataset, task)
    """

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

    # Cross-subject classification
    result_file_cross = "../results/cross-subj-zucoAll.csv"
    results_cross = pd.read_csv(result_file_cross, delimiter=" ", names=["subject", "feature_set", "accuracy", "samples", "run"])
    dataset = result_file_cross.split("-")[2].replace(".csv", "")
    #print(dataset)
    #cross_subj_results(results_cross, dataset)

    # Fixation order results
    #result_file_fix = "../results/2021-07-30_svm_all_runs_tasks_zuco1_randomFalse_linear_FixFeats.csv"
    result_file_fix = "../results/2021-07-30_svm_all_runs_tasks_zuco2_randomFalse_linear_FixFeats.csv"
    results_fix = pd.read_csv(result_file_fix, delimiter=" ",
                                names=["subject", "feature_set", "accuracy", "samples", "run"])
    dataset = result_file_fix.split("_")[5]
    print(dataset)
    plot_results_fixations(results_fix, dataset)



if __name__ == '__main__':
    main()

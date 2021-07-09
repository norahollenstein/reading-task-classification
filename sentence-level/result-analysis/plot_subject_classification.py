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
    plt.title("Subject classification")
    plt.tight_layout()
    plt.savefig("plots/subj_class_"+dataset+".pdf")
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
    result_file = "../results/2021-06-16_svm_averaged_results_subjects_zuco1_randomFalse_linear.csv"
    results = pd.read_csv(result_file, delimiter=" ", names=["subject", "feature_set", "accuracy", "std"])
    plot_results_detailed(results, "zuco1")




if __name__ == '__main__':
    main()

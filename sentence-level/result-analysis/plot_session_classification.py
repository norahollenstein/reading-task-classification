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

    for f in features:
        feature_results = results[results['feature_set'] == f]
        feature_results = feature_results.sort_values(by=['accuracy'])

        print("Mean accuracy:", f, np.mean(feature_results['accuracy']))

    ax = sns.barplot(x=results["feature_set"], y=results["accuracy"], palette=sns.color_palette("Spectral", len(features)))
    ax.set_xticklabels(results["feature_set"], rotation=90)
    plt.title("Session classification")
    plt.tight_layout()
    plt.savefig("plots/session_class_"+dataset+".pdf")
    plt.show()


def plot_results_compared(results_sess, results_tasks, dataset):
    features = results_sess.feature_set.unique()
    print(features)
    print(len(results_sess))
    results_sess = results_sess.drop_duplicates()
    print(len(results_sess))

    print(len(results_tasks))
    results_tasks = results_tasks.drop_duplicates()
    print(len(results_tasks))

    for f in features:
        feature_results_sess = results_sess[results_sess['feature_set'] == f]
        feature_results_sess = feature_results_sess.sort_values(by=['accuracy'])

        print("Mean accuracy:", f, np.mean(feature_results_sess['accuracy']))

    plt.figure()
    sns.barplot(x=results_sess["feature_set"], y=results_sess["accuracy"],
                     palette=sns.color_palette("Spectral", len(features)))
    sns.barplot(x=results_tasks["feature_set"], y=results_tasks["accuracy"],
                     palette=sns.color_palette("viridis", len(features)))

    #ax2.set_xticklabels(results_sess["feature_set"], rotation=90)
    plt.title("Session classification")
    plt.tight_layout()
    plt.savefig("plots/session_class_" + dataset + ".pdf")
    plt.show()



def main():
    result_file_sessions = "../results/2021-07-13_svm_results_sessions_zuco1sr_randomFalse_linear.csv"
    results = pd.read_csv(result_file_sessions, delimiter=" ", names=["subject", "feature_set", "accuracy", "std", "features", "samples", "runs"])
    print(results.head())
    dataset = result_file_sessions.split("_")[4]
    plot_results_detailed(results, dataset)

    result_file_tasks = "../results/tasks-zuco1-final-avg.csv"
    results_tasks = pd.read_csv(result_file_sessions, delimiter=" ", names=["subject", "feature_set", "accuracy", "std", "features", "samples", "runs"])
    print(results_tasks.head())
    dataset = result_file_sessions.split("-")[1]
    plot_results_compared(results, results_tasks, dataset)




if __name__ == '__main__':
    main()

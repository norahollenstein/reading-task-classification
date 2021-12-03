import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def plot_results_blocks(results_all, dataset):
    features = results_all.feature_set.unique()
    print(features)

    with sns.plotting_context(font_scale=10):
        ax = sns.lineplot(x="blocks", y="accuracy", data=results_all, hue="feature_set")
        #plt.title("ZuCo 2.0 - Reading task classification")
        plt.xlabel("blocks per task", fontsize=16)
        plt.ylabel("accuracy", fontsize=16)
        plt.tight_layout()
        handles, labels = ax.get_legend_handles_labels()
        # sort both labels and handles by labels
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
        ax.legend(handles, labels)
        plt.savefig("plots/task_class_blocks_"+dataset+".pdf")
        plt.show()


def main():

    results_all = pd.DataFrame()
    for n in list(range(1,7)):
        result_file = "../results/2021-07-21_svm_results_blocks-in-sets_zuco2_randomFalse_linear_"+str(n)+".csv"
        print(result_file)
        results = pd.read_csv(result_file, delimiter=" ", names=["subject", "feature_set", "accuracy", "std", "features", "samples", "runs"])
        print(len(results))
        results["blocks"] = n
        print(results)
        results_all = results_all.append(results, ignore_index=True)
        dataset = "zuco2"
    #print(results_all)
    plot_results_blocks(results_all, dataset)




if __name__ == '__main__':
    main()

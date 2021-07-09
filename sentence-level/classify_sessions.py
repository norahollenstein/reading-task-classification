import numpy as np
from sklearn.metrics import classification_report
import extract_features as fe
import classifier
import config
import h5py
import time
import data_helpers as dh
from datetime import timedelta, date

# classify sentences into session 1 and session 2 for each subject of ZuCo 1 separately (with or without SR sentences)

def main():

    start = time.time()

    #result_file = open("results/"+str(date.today())+"_svm_detailed_results_"+config.class_task+"_"+config.dataset+"_random"+str(config.randomized_labels)+".csv", "w")
    #avg_result_file = open("results/"+str(date.today())+"_svm_averaged_results_"+config.class_task+"_"+config.dataset+"_random"+str(config.randomized_labels)+".csv", "w")
    #subj_avg_result_file = open("results/"+str(date.today())+"_svm_subject_results_"+config.class_task+"_"+config.dataset+"_random"+str(config.randomized_labels)+".csv", "w")

    subj_result_file, all_runs_result_file, coef_file = dh.prepare_output_files()

    avg_results = {}
    for subject in config.subjects:
        print(subject)
        filename_nr = config.rootdir + "results" + subject + "_NR.mat"
        filename_tsr = config.rootdir + "results" + subject + "_TSR.mat"

        f_nr = dh.read_mat_file(filename_nr)
        f_tsr = dh.read_mat_file(filename_tsr)

        if config.dataset is "zuco1_sr":  # include sentiment reading as NR
            filename_sr = config.rootdir + "results" + subject + "_SR.mat"
            f_sr = dh.read_mat_file(filename_sr)

        features = {}

        for feature_set in config.feature_sets:

            features[feature_set] = {}

            fe.extract_sentence_features(subject, f_nr, feature_set, features, "Sess1")
            fe.extract_sentence_features(subject, f_tsr, feature_set, features, "Sess2")
            if config.dataset is "zuco1sr":
                fe.extract_sentence_features(subject, f_sr, feature_set, features, "SR-Sess")
            print(len(features[feature_set]), " samples collected for", feature_set)
            print(features.keys())

            accuracies = []; predictions = []; true_labels = []
            for i in range(config.runs):
                preds, test_y, acc = classifier.svm_sessions(features[feature_set], config.seed+i, config.randomized_labels)

                accuracies.append(acc)
                predictions.extend(preds)
                true_labels.extend(test_y)

            # detailed results: subject name, feature set, mean accuracy, no. of features, no. of samples
            report = classification_report(true_labels, predictions, labels=[0, 1], target_names=["Sess1", "Sess2"], output_dict=True)
            print(subject, feature_set, np.mean(accuracies), report["Sess1"]['f1-score'], report["Sess2"]['f1-score'], len(features[feature_set][list(features[feature_set].keys())[0]])-1, len(features[feature_set]), file=result_file)

            subj_results.append(np.mean(accuracies))

            if feature_set in avg_results:
                avg_results[feature_set].append(np.mean(accuracies))
            else:
                avg_results[feature_set] = [np.mean(accuracies)]

        # print average accuracy per subject over all feature sets
        print(subject, np.mean(subj_results), file=subj_avg_result_file)
        print(subject, np.mean(subj_results))

    for feat_set, results in avg_results.items():
        print(feat_set, np.mean(results), np.std(results), file=avg_result_file)

    elapsed = (time.time() - start)
    print(str(timedelta(seconds=elapsed)))


if __name__ == '__main__':
    main()

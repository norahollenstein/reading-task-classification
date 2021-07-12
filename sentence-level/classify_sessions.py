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

    subj_result_file, all_runs_result_file, coef_file = dh.prepare_output_files()

    avg_results = {}
    for subject in config.subjects:
        print(subject)
        filename_nr = config.rootdir + "results" + subject + "_NR.mat"
        filename_tsr = config.rootdir + "results" + subject + "_TSR.mat"

        f_nr = dh.read_mat_file(filename_nr)
        f_tsr = dh.read_mat_file(filename_tsr)

        if config.dataset is "zuco1sr":  # include sentiment reading as NR
            filename_sr = config.rootdir_sr + "results" + subject + "_SR.mat"
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

            # print(features[feature_set])
            dh.plot_feature_distribution(subject, config.dataset, features[feature_set], feature_set)

            predictions = [];
            true_labels = [];
            accuracies = [];
            svm_coeffs = []
            for i in range(config.runs):
                # print(i)
                preds, test_y, acc, coefs = classifier.svm(features[feature_set], config.seed + i,
                                                           config.randomized_labels)

                accuracies.append(acc)
                predictions.extend(preds)
                true_labels.extend(test_y)
                svm_coeffs.append(coefs[0])

                # print results of each run
                print(subject, feature_set, acc, len(features[feature_set]), i, file=all_runs_result_file)

            avg_svm_coeffs = np.mean(np.array(svm_coeffs), axis=0)

            # print SVM coefficients to file
            print(subject, feature_set, " ".join(map(str, avg_svm_coeffs)), file=coef_file)

            # print results for individual subjects to file
            print("Classification accuracy:", subject, feature_set, np.mean(accuracies), np.std(accuracies))
            # subj, feature set, acc, std, no. of feature, no. of samples, no. of runs
            print(subject, feature_set, np.mean(accuracies), np.std(accuracies),
                  len(features[feature_set][list(features[feature_set].keys())[0]]) - 1, len(features[feature_set]),
                  config.runs, file=subj_result_file)

    elapsed = (time.time() - start)
    print(str(timedelta(seconds=elapsed)))



if __name__ == '__main__':
    main()

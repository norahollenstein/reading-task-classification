import numpy as np
import extract_features as fe
import classifier
import config
import data_helpers as dh
import time
from datetime import timedelta
from sklearn.metrics import confusion_matrix

# classify experiments blocks of ZuCo 2

# Recording sequence
# NR_1 TSR_1 NR_2 | TSR_2 NR_3 TSR_3 NR_4 | TSR_4 NR_5 TSR_5 NR_6 | TSR_6 NR_7 TSR_7
# „|“ marks breaks for EEG impedance testing.
# Number of sentences (excluding practice sentences):
# 50 45 50 | 72 51 54 50 | 65 50 54 49 | 60 49 40
# Total: 739


def main():

    start = time.time()

    subj_result_file, all_runs_result_file, coef_file = dh.prepare_output_files()

    all_true = []; all_predictions = []

    for subject in config.subjects:
        print(subject)
        filename_nr = config.rootdir + "results" + subject + "_NR.mat"
        filename_tsr = config.rootdir + "results" + subject + "_TSR.mat"

        f_nr = dh.read_mat_file(filename_nr)
        f_tsr = dh.read_mat_file(filename_tsr)

        features = {}

        for feature_set in config.feature_sets:

            features[feature_set] = {}

            fe.extract_sentence_features(subject, f_nr, feature_set, features, "NR")
            print(len(features[feature_set]), " NR samples collected for", feature_set)
            fe.extract_sentence_features(subject, f_tsr, feature_set, features, "TSR")
            print(len(features[feature_set]), " total samples collected for", feature_set)

            #print(features[feature_set])
            #dh.plot_feature_distribution(subject, config.dataset, features[feature_set], feature_set)

            #print(features[feature_set])

            predictions = []; true_labels = []; accuracies = []; svm_coeffs = []
            for i in range(config.runs):
                #print(i)
                preds, test_y, acc, coefs = classifier.svm(features[feature_set], config.seed+i, config.randomized_labels)

                accuracies.append(acc)
                predictions.extend(preds)
                all_predictions.extend(preds)
                true_labels.extend(test_y)
                all_true.extend(test_y)
                svm_coeffs.append(coefs[0])

                # print results of each run
                print(subject, feature_set, acc, len(features[feature_set]), i, file=all_runs_result_file)

            avg_svm_coeffs = np.mean(np.array(svm_coeffs), axis=0)


            # print SVM coefficients to file
            print(subject, feature_set, " ".join(map(str, avg_svm_coeffs)), file=coef_file)

            # print results for individual subjects to file
            print("Classification accuracy:", subject, feature_set, np.mean(accuracies), np.std(accuracies))
            # subj, feature set, acc, std, no. of feature, no. of samples, no. of runs
            print(subject, feature_set, np.mean(accuracies), np.std(accuracies), len(features[feature_set][list(features[feature_set].keys())[0]])-1, len(features[feature_set]), config.runs, file=subj_result_file)

    cm = confusion_matrix(true_labels, predictions)
    print(cm)
    target_names = ["NR_1", "TSR_1", "NR_2", "TSR_2", "NR_3", "TSR_3", "NR_4",
              "TSR_4", "NR_5", "TSR_5", "NR_6", "TSR_6", "NR_7", "TSR_7"]
    dh.multi_conf_matrix(target_names, config.feature_sets[0], cm)

    elapsed = (time.time() - start)
    print(str(timedelta(seconds=elapsed)))


if __name__ == '__main__':
    main()

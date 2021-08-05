import numpy as np
from sklearn.metrics import classification_report
import extract_features as fe
import classifier
import config
import h5py
import time
import data_helpers as dh
from datetime import timedelta


# classify NR vs TSR, leave-one-out cross-subject. train on all-1 subjects, test on left out subject


def main():

    start = time.time()

    subj_result_file, all_runs_result_file, coef_file = dh.prepare_output_files()

    features = {}

    for subject in config.subjects:
        print(subject)

        if subject.startswith("Z"):
            filename_nr = config.rootdir + "results" + subject + "_NR.mat"
            filename_tsr = config.rootdir + "results" + subject + "_TSR.mat"
            #filename_sr = config.rootdir1 + "results" + subject + "_SR.mat"

        elif subject.startswith("Y"):
            filename_nr = config.rootdir + "results" + subject + "_NR.mat"
            filename_tsr = config.rootdir + "results" + subject + "_TSR.mat"

        f_nr = dh.read_mat_file(filename_nr)
        f_tsr = dh.read_mat_file(filename_tsr)

        for feature_set in config.feature_sets:
            #print(feature_set)

            if feature_set not in features:
                features[feature_set] = {}

            fe.extract_sentence_features(subject, f_nr, feature_set, features, "NR")
            fe.extract_sentence_features(subject, f_tsr, feature_set, features, "TSR")
            #if subject.startswith("Z"):
             #   f_sr = dh.read_mat_file(filename_sr)
              #  fe.extract_sentence_features(subject, f_sr, feature_set, features, "NR")

    print(len(features[feature_set]), " samples collected for", feature_set)

    dh.plot_feature_distribution("ALL", config.dataset, features[feature_set], feature_set)

    for subject in config.subjects:
        for set, feats in features.items():
            accuracies = []; predictions = []; true_labels = []; svm_coeffs = []
            print("\nTraining on all subjects, testing on", subject)
            for i in range(config.runs):
                preds, test_y, acc, coefs = classifier.svm_cross_subj(feats, config.seed+i, subject, config.randomized_labels)
                accuracies.append(acc)
                predictions.extend(preds)
                true_labels.extend(test_y)
                svm_coeffs.append(coefs[0])

                # print results of each run
                print(subject, feature_set, acc, len(features[feature_set]), i, file=all_runs_result_file)

        avg_svm_coeffs = np.mean(np.array(svm_coeffs), axis=0)

        # print SVM coefficients to fil
        print(subject, feature_set, " ".join(map(str, avg_svm_coeffs)), file=coef_file)

        # print results for individual subjects to file
        print("Classification accuracy:", subject, feature_set, np.mean(accuracies), np.std(accuracies))
        print(subject, feature_set, np.mean(accuracies), np.std(accuracies),
              len(features[feature_set][list(features[feature_set].keys())[0]]) - 1, len(features[feature_set]),
              file=subj_result_file)

    elapsed = (time.time() - start)
    print(str(timedelta(seconds=elapsed)))


if __name__ == '__main__':
    main()

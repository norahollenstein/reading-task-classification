import numpy as np
import extract_features as fe
import classifier
import config
import h5py
import time
import data_helpers as dh
from datetime import timedelta
from datetime import timedelta, date


def main():

    start = time.time()

    # todo: fix this!
    subj_result_file, all_runs_result_file, coef_file = dh.prepare_output_files()

    features = {}

    for subject in config.subjects:
        print(subject)

        filename_nr = config.rootdir + "results" + subject + "_NR.mat"
        filename_tsr = config.rootdir + "results" + subject + "_TSR.mat"

        f_nr = dh.read_mat_file(filename_nr)
        f_tsr = dh.read_mat_file(filename_tsr)

        for feature_set in config.feature_sets:

            if feature_set not in features:
                features[feature_set] = {}

            fe.extract_sentence_features(subject, f_nr, feature_set, features, subject)
            fe.extract_sentence_features(subject, f_tsr, feature_set, features, subject)
            print(len(features[feature_set]), " samples collected for", feature_set)

    for set, feats in features.items():
        accuracies = []; predictions = []; true_labels = []
        print("\nTraining models for", set)
        for i in range(config.runs):
            preds, test_y, acc, coefs = classifier.svm_cross_subj(features[set], config.seed + i, config.randomized_labels)

            accuracies.append(acc)
            predictions.extend(preds)
            true_labels.extend(test_y)

        print("allSubjects", set, np.mean(accuracies))
        print("allSubjects", set, np.mean(accuracies), np.std(accuracies), file=avg_result_file)

    elapsed = (time.time() - start)
    print(str(timedelta(seconds=elapsed)))


if __name__ == '__main__':
    main()

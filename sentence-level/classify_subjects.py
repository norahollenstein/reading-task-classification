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
    avg_result_file = open("results/"+str(date.today())+"_svm_averaged_results_"+config.class_task+"_"+config.dataset+"_random"+str(config.randomized_labels)+ "_" + config.kernel + ".csv", "a")

    features = {}

    for subject in config.subjects:
        print(subject)

        print(subject)
        filename_nr = config.rootdir + "results" + subject + "_NR.mat"
        filename_tsr = config.rootdir + "results" + subject + "_TSR.mat"

        f_nr = dh.read_mat_file(filename_nr)
        f_tsr = dh.read_mat_file(filename_tsr)

        if config.dataset is "zuco1_sr":  # include sentiment reading as NR
            filename_sr = config.rootdir + "results" + subject + "_SR.mat"
            f_sr = dh.read_mat_file(filename_sr)


        for feature_set in config.feature_sets:
            #print(feature_set)

            if feature_set not in features:
                features[feature_set] = {}

            fe.extract_sentence_features(subject, f_nr, feature_set, features, subject)
            fe.extract_sentence_features(subject, f_tsr, feature_set, features, subject)
            if config.dataset is "zuco1_sr":
                fe.extract_sentence_features(subject, f_sr, feature_set, features, subject)
            print(len(features[feature_set]), " samples collected for", feature_set)


    for set, feats in features.items():
        accuracies = []; predictions = []; true_labels = []
        print("\nTraining models for", set)
        #print(feats)
        for i in range(config.runs):
            preds, test_y, acc, coefs = classifier.svm(features[set], config.seed + i, config.randomized_labels)

            accuracies.append(acc)
            predictions.extend(preds)
            true_labels.extend(test_y)

        print("allSubjects", set, np.mean(accuracies))
        print("allSubjects", set, np.mean(accuracies), np.std(accuracies), file=avg_result_file)

    elapsed = (time.time() - start)
    print(str(timedelta(seconds=elapsed)))


if __name__ == '__main__':
    main()

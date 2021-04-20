import numpy as np
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.metrics import classification_report
import extract_features as fe
import classifier
import config
import h5py
import time
from datetime import timedelta

# classify NR vs TSR, average feature values over all subjects


def main():
    """
    start = time.time()

    result_file = open("svm_detailed_results_"+config.class_task+"_"+config.dataset+"_2_allSubjs.csv", "w")

    features = {}

    for subject in config.subjects:
        print(subject)

        if subject.startswith("Z"):
            filename_nr = config.rootdir1 + "results" + subject + "_NR.mat"
            filename_tsr = config.rootdir1 + "results" + subject + "_TSR.mat"
            filename_sr = config.rootdir1 + "results" + subject + "_SR.mat"

        elif subject.startswith("Y"):
            filename_nr = config.rootdir2 + "results" + subject + "_NR.mat"
            filename_tsr = config.rootdir2 + "results" + subject + "_TSR.mat"

        f_nr = h5py.File(filename_nr, 'r')
        sentence_data_nr = f_nr['sentenceData']

        f_tsr = h5py.File(filename_tsr, 'r')
        sentence_data_tsr = f_tsr['sentenceData']

        for feature_set in config.feature_sets:
            #print(feature_set)

            if feature_set not in features:
                features[feature_set] = {}

            fe.extract_sentence_features(subject, f_nr, sentence_data_nr, feature_set, features, "NR")
            fe.extract_sentence_features(subject, f_tsr, sentence_data_tsr, feature_set, features, "TSR")
            if subject.startswith("Z"):
                f_sr = h5py.File(filename_sr, 'r')
                sentence_data_sr = f_sr['sentenceData']
                fe.extract_sentence_features(subject, f_sr, sentence_data_sr, feature_set, features, "NR")

            print(len(features[feature_set]), " samples collected for", feature_set)
    """

    features = {'omission_rate' : {'ZAB_NR_1': [0.4, "NR"], 'YTL_NR_1': [0.6, "NR"]}}

    for set, feats in features.items():
        # average feature values for each sentence over all subjects
        print(set)
        features_avg = {}
        for sample_id, value in feats.items():
            sent_id = sample_id.partition("_")[2]
            #print(sent_id)
            # todo: fix for mutliple feature values
            if sent_id not in features_avg:
                features_avg[sent_id] = [[value[0]], value[1]]
            else:
                features_avg[sent_id][0].append(value[0])

        print(features_avg)

        for id, vals in features_avg.items():
            print(id, vals)
            features_avg[id][0] = np.mean(vals[0])

        print(features_avg)

    """
        accuracies = []; predictions = []; true_labels = []
        print("Training models for", set)
        for i in range(100):
            preds, test_y, acc = classifier.svm(features_avg, config.seed+i)
            accuracies.append(acc)
            predictions.extend(preds)
            true_labels.extend(test_y)

        # detailed results: subject name, feature set, mean accuracy, no. of features, no. of samples
        report = classification_report(true_labels, predictions, labels=[0, 1], target_names=["TSR", "NR"], output_dict=True)

        print("allSubjects", set, np.mean(accuracies), report["NR"]['f1-score'], report["TSR"]['f1-score'], len(list(feats.keys())[0])-1, len(feats), file=result_file)
        print("\nallSubjects", set, np.mean(accuracies))

    elapsed = (time.time() - start)
    print(str(timedelta(seconds=elapsed)))
    """


if __name__ == '__main__':
    main()

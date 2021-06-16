import numpy as np
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.metrics import classification_report
import sklearn.metrics
import extract_features as fe
import classifier
import config
import h5py
import time
from datetime import timedelta
from plot_confusion_matrix import multi_conf_matrix
from datetime import timedelta, date


def main():

    start = time.time()

    avg_result_file = open("results/"+str(date.today())+"_svm_averaged_results_"+config.class_task+"_"+config.dataset+"_random"+str(config.randomized_labels)+ "_" + config.kernel + ".csv", "a")

    features = {}

    for subject in config.subjects:
        print(subject)

        if subject.startswith("Z"):
            filename_nr = config.rootdir1 + "results" + subject + "_NR.mat"
            filename_tsr = config.rootdir1 + "results" + subject + "_TSR.mat"
            #filename_sr = config.rootdir1 + "results" + subject + "_SR.mat"

        elif subject.startswith("Y"):
            filename_nr = config.rootdir + "results" + subject + "_NR.mat"
            filename_tsr = config.rootdir + "results" + subject + "_TSR.mat"

        f_nr = h5py.File(filename_nr, 'r')
        sentence_data_nr = f_nr['sentenceData']

        f_tsr = h5py.File(filename_tsr, 'r')
        sentence_data_tsr = f_tsr['sentenceData']

        for feature_set in config.feature_sets:
            #print(feature_set)

            if feature_set not in features:
                features[feature_set] = {}

            fe.extract_sentence_features(subject, f_nr, sentence_data_nr, feature_set, features, subject)
            fe.extract_sentence_features(subject, f_tsr, sentence_data_tsr, feature_set, features, subject)
            #if subject.startswith("Z"):
             #   f_sr = h5py.File(filename_sr, 'r')
             #   sentence_data_sr = f_sr['sentenceData']
             #   fe.extract_sentence_features(subject, f_sr, sentence_data_sr, feature_set, features, subject)

            print(len(features[feature_set]), " samples collected for", feature_set)

    for set, feats in features.items():
        accuracies = []; predictions = []; true_labels = []
        print("\nTraining models for", set)
        print(feats)
        for i in range(100):
            preds, test_y, acc = classifier.svm_multiclass(feats, config.seed+i)
            accuracies.append(acc)
            predictions.extend(preds)
            true_labels.extend(test_y)

        #report = classification_report(true_labels, predictions, labels=list(range(len(config.subjects))), target_names=config.subjects, output_dict=True)

        conf_matrix = sklearn.metrics.confusion_matrix(true_labels, predictions, labels=list(range(len(config.subjects))))
        print(conf_matrix)
        multi_conf_matrix(config.subjects, set, conf_matrix)

        #for sub in config.subjects:
         #   print(sub, set, report[sub]['f1-score'], len(feats[list(feats.keys())[0]])-1, len(feats), file=result_file)
         #   print(sub, set, report[sub]['f1-score'], len(feats[list(feats.keys())[0]]) - 1, len(feats))

        #print(set, np.mean(accuracies), np.std(accuracies), file=all_runs)

        print("allSubjects", set, np.mean(accuracies))
        print("allSubjects", set, np.mean(accuracies), np.std(accuracies), file=avg_result_file)

    elapsed = (time.time() - start)
    print(str(timedelta(seconds=elapsed)))


if __name__ == '__main__':
    main()

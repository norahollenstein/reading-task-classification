import numpy as np
from sklearn.metrics import classification_report
import extract_features as fe
import classifier
import config
import h5py
import time
from datetime import timedelta, date

# classify NR vs TSR for each subject separately.
# features: average EEG signals only during fixations falling inside the wordbounds

def main():

    start = time.time()

    result_file = open("results/"+str(date.today())+"_svm_detailed_results_"+config.class_task+"_"+config.dataset+"_random"+str(config.randomized_labels)+"-wordFix.csv", "a")
    avg_result_file = open("results/"+str(date.today())+"_svm_averaged_results_"+config.class_task+"_"+config.dataset+"_random"+str(config.randomized_labels)+"-wordFix.csv", "a")
    #subj_avg_result_file = open("results/"+str(date.today())+"_svm_subject_results_"+config.class_task+"_"+config.dataset+"_random"+str(config.randomized_labels)+"-test.csv", "a")

    avg_results = {};
    for subject in config.subjects:
        print(subject)
        subj_results = []
        filename_nr = config.rootdir + "results" + subject + "_NR.mat"
        filename_tsr = config.rootdir + "results" + subject + "_TSR.mat"

        f_nr = h5py.File(filename_nr, 'r')
        sentence_data_nr = f_nr['sentenceData']

        f_tsr = h5py.File(filename_tsr, 'r')
        sentence_data_tsr = f_tsr['sentenceData']

        if config.dataset is "zuco1_sr":  # include sentiment reading as NR
            filename_sr = config.rootdir + "results" + subject + "_SR.mat"
            f_sr = h5py.File(filename_sr, 'r')
            sentence_data_sr = f_sr['sentenceData']

        features = {}

        for feature_set in config.feature_sets:
            print(feature_set)

            features[feature_set] = {}

            fe.extract_fixation_features(subject, f_nr, sentence_data_nr, feature_set, features, "NR")
            fe.extract_fixation_features(subject, f_tsr, sentence_data_tsr, feature_set, features, "TSR")
            if config.dataset is "zuco1_sr":
                fe.extract_fixation_features(subject, f_sr, sentence_data_sr, feature_set, features, "NR")
            print(len(features[feature_set]), " samples collected for", feature_set)
            #print(features.keys())

            accuracies = []; predictions = []; true_labels = []

            #print(features[feature_set])

            for i in range(100):
                preds, test_y, acc, coefs = classifier.svm(features[feature_set], config.seed+i, config.randomized_labels)

                accuracies.append(acc)
                predictions.extend(preds)
                true_labels.extend(test_y)

            # detailed results: subject name, feature set, mean accuracy, no. of features, no. of samples
            report = classification_report(true_labels, predictions, labels=[0, 1], target_names=["TSR", "NR"], output_dict=True)
            print(subject, feature_set, np.mean(accuracies), report["NR"]['f1-score'], report["TSR"]['f1-score'], len(features[feature_set][list(features[feature_set].keys())[0]])-1, len(features[feature_set]), file=result_file)

            print(report)
            subj_results.append(np.mean(accuracies))

            if feature_set in avg_results:
                avg_results[feature_set].append(np.mean(accuracies))
            else:
                avg_results[feature_set] = [np.mean(accuracies)]

        # print average accuracy per subject over all feature sets
        #print(subject, np.mean(subj_results), file=subj_avg_result_file)
        #print(subject, np.mean(subj_results))

    for feat_set, results in avg_results.items():
        print(feat_set, results)
        print(feat_set, np.mean(results), np.std(results), file=avg_result_file)

    elapsed = (time.time() - start)
    print(str(timedelta(seconds=elapsed)))


if __name__ == '__main__':
    main()

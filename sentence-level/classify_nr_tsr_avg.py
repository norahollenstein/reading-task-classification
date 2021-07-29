import numpy as np
import extract_features as fe
import config
import data_helpers as dh
import time
from datetime import timedelta
import pandas as pd
import mne
from mne import EvokedArray
import matplotlib.pyplot as plt


# classify NR vs TSR for each subject separately


def main():

    start = time.time()

    subj_result_file, all_runs_result_file, coef_file = dh.prepare_output_files()

    features_all = pd.DataFrame(columns=['subj', 'feature_set', 'sample_id', 'feature_values', 'label'])

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

            fe.extract_sentence_features(subject, f_nr, feature_set, features, "NR")
            fe.extract_sentence_features(subject, f_tsr, feature_set, features, "TSR")
            if config.dataset is "zuco1_sr":
                fe.extract_sentence_features(subject, f_sr, feature_set, features, "NR")
            print(len(features[feature_set]), " samples collected for", feature_set)

            for x, y in features[feature_set].items():
                features_all = features_all.append({'subj': subject, 'feature_set': feature_set, 'sample_id': x, 'feature_values': np.array(y[:-1]), 'label':y[-1]}, ignore_index=True)

    print(features_all.head)

    #features_avg = pd.DataFrame(columns=['subj', 'feature_set', 'sample_id', 'feature_values', 'label'])


    for feature_set in config.feature_sets:

        # print(features[feature_set])
        info = mne.create_info(ch_names=config.chanlocs, ch_types="bio", sfreq=500)

        features_nr = features_all.loc[(features_all['feature_set'] == feature_set) & (features_all['label'] == 'NR')]
        mean_nr = features_nr['feature_values'].mean()
        print(mean_nr)

        evoked = EvokedArray(mean_nr.reshape(-1,1), info=info)
        evoked.set_montage("GSN-HydroCel-128")

        fig, ax = plt.subplots(figsize=(7.5, 4.5), nrows=1, ncols=1)
        ax = evoked.plot_topomap(title='EEG patterns', time_unit='s')
        plt.savefig("NR-topo-AVG-ALL.pdf")

        features_tsr = features_all.loc[(features_all['feature_set'] == feature_set) & (features_all['label'] == 'TSR')]
        mean_tsr = features_tsr['feature_values'].mean()
        print(mean_tsr)

        diff = mean_nr - mean_tsr


        #print(features_avg)

        #x = features_all.groupby(['feature_set', 'label'], as_index=False).mean()
        #print(x)

        """
        dh.plot_feature_distribution("AVG", config.dataset, features_all[feature_set], feature_set)

        predictions = [];
        true_labels = [];
        accuracies = [];
        svm_coeffs = []
        for i in range(config.runs):
            # print(i)
            preds, test_y, acc, coefs = classifier.svm(features[feature_set], config.seed + i, i,
                                                       config.randomized_labels)

            accuracies.append(acc)
            predictions.extend(preds)
            true_labels.extend(test_y)
            svm_coeffs.append(coefs[0])

            # print results of each run
            print(subject, feature_set, acc, len(features[feature_set]), i, file=all_runs_result_file)
        """

    #dh.plot_feature_distribution(subject, config.dataset, features_all[feature_set], feature_set)


    elapsed = (time.time() - start)
    print(str(timedelta(seconds=elapsed)))


if __name__ == '__main__':
    main()

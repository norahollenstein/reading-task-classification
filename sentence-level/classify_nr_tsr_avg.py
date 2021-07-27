import numpy as np
import extract_features as fe
import classifier
import config
import data_helpers as dh
import time
from datetime import timedelta
import pandas as pd

# classify NR vs TSR for each subject separately


def main():

    start = time.time()

    subj_result_file, all_runs_result_file, coef_file = dh.prepare_output_files()

    features_all = pd.DataFrame()

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
                features_all.append([x, y])

        print(features_all)

    elapsed = (time.time() - start)
    print(str(timedelta(seconds=elapsed)))


if __name__ == '__main__':
    main()

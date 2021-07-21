import config
from feature_extraction import zuco_reader
from models import eeg_model
from data_helpers import save_results, load_matlab_files
import collections
import json
import numpy as np
import sys

# Usage on spaceml:
# $ conda activate env-nrtsr
# $ CUDA_VISIBLE_DEVICES=7 python tune_model.py


def main():

    print("TASK: ", config.class_task)
    print("Extracting", config.feature_set, "features....")
    train_feature_dict = {}
    train_label_dict = {}
    train_eeg_dict = {}
    train_gaze_dict = {}
    test_feature_dict = {}
    test_label_dict = {}
    test_eeg_dict = {}
    test_gaze_dict = {}
    for subject in config.subjects:

        if subject not in ["ZDN", "ZJN", "ZPH"]:

            train_subjs = [x for x in config.subjects if x != subject]
            test_subj = subject
            print(train_subjs, test_subj)

            for tr_subj in train_subjs:
                loaded_data = load_matlab_files(config.class_task, tr_subj)
                zuco_reader.extract_features(loaded_data, config.feature_set, train_feature_dict, train_eeg_dict, train_gaze_dict, train_label_dict)
                print(len(train_feature_dict), len(train_label_dict), len(train_eeg_dict))

            loaded_data = load_matlab_files(config.class_task, test_subj)
            zuco_reader.extract_features(loaded_data, config.feature_set, test_feature_dict, test_eeg_dict,
                                         test_gaze_dict, test_label_dict)
            print(len(test_feature_dict), len(test_label_dict), len(test_eeg_dict))

            """
            if config.run_feature_extraction:
                # saving gaze features to file
                print(len(gaze_dict))
                with open("features/" + subject + "_" + config.feature_set[
                    0] + '_feats_file_' + config.class_task + '_Sacc' + str(config.saccades) + '.json', 'w') as fp:
                    json.dump(gaze_dict, fp)
                print("saved.")
            else:
                gaze_dict = json.load(open("features/" + subject + "_" + config.feature_set[
                    0] + '_feats_file_' + config.class_task + '_Sacc' + str(config.saccades) + '.json'))
            """

            print(len(train_feature_dict), len(train_label_dict), len(train_eeg_dict))
            if len(train_feature_dict) != len(train_label_dict) or len(train_feature_dict) != len(train_eeg_dict):
                print("WARNING: Not an equal number of sentences in features and labels!")

            train_feature_dict = collections.OrderedDict(sorted(train_feature_dict.items()))
            train_label_dict = collections.OrderedDict(sorted(train_label_dict.items()))
            train_gaze_dict = collections.OrderedDict(sorted(train_eeg_dict.items()))

            # eliminate sentence without available eye-tracking features
            for sent, feats in list(train_label_dict.items()):
                if sent not in train_eeg_dict:
                    del train_label_dict[sent]
                    del train_feature_dict[sent]

            for rand in config.random_seed_values:
                np.random.seed(rand)
                for lstmDim in config.lstm_dim:
                    for lstmLayers in config.lstm_layers:
                        for denseDim in config.dense_dim:
                            for drop in config.dropout:
                                for bs in config.batch_size:
                                    for lr_val in config.lr:
                                        for e_val in config.epochs:
                                            parameter_dict = {"lr": lr_val, "lstm_dim": lstmDim, "lstm_layers": lstmLayers,
                                                              "dense_dim": denseDim, "dropout": drop, "batch_size": bs,
                                                              "epochs": e_val, "random_seed": rand}

                                            if config.class_task == "tasks-cross-subj":
                                                fold_results = eeg_model.lstm_classifier_cross(train_label_dict, train_gaze_dict, test_label_dict, test_eeg_dict,
                                                                                                    parameter_dict, rand)
                                                save_results(fold_results, config.class_task, subject)


if __name__ == '__main__':
    main()

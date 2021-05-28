import config
from feature_extraction import zuco_reader
from models import gaze_model
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
    for subject in config.subjects:
        print(subject)
        feature_dict = {}
        label_dict = {}
        eeg_dict = {}
        gaze_dict = {}

        loaded_data = load_matlab_files(config.class_task, subject)

        zuco_reader.extract_features(loaded_data, config.feature_set, feature_dict, eeg_dict, gaze_dict, label_dict)

        print(len(feature_dict), len(label_dict), len(gaze_dict))

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

        print(len(feature_dict), len(label_dict), len(gaze_dict))
        if len(feature_dict) != len(label_dict) != len(gaze_dict):
            print("WARNING: Not an equal number of sentences in features and labels!")

        feature_dict = collections.OrderedDict(sorted(feature_dict.items()))
        label_dict = collections.OrderedDict(sorted(label_dict.items()))
        gaze_dict = collections.OrderedDict(sorted(gaze_dict.items()))
        for sent, feats in list(label_dict.items()):
            if sent not in gaze_dict:
                print(sent)
                del label_dict[sent]
                del feature_dict[sent]

        print(len(feature_dict), len(label_dict), len(gaze_dict))
        if len(feature_dict) != len(label_dict) != len(gaze_dict):
            print("WARNING: Not an equal number of sentences in features and labels!")

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

                                        if config.class_task == "read-task":
                                            fold_results = gaze_model.lstm_classifier(label_dict, gaze_dict,
                                                                                                parameter_dict, rand)
                                            save_results(fold_results, config.class_task, subject)


if __name__ == '__main__':
    main()

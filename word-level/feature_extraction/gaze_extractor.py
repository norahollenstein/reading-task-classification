from . import data_loading_helpers as dh
import numpy as np
import config


def word_level_et_features(sentence_data, gaze_dict, label_dict):
    """extract word level eye-tracking features from Matlab files"""

    for tup in sentence_data:

        # read Matlab v7.3 structures
        f = tup[0]
        s_data = tup[1]
        if tup[2] == "NR":
            label = 0
        elif tup[2] == "TSR":
            label = 1
        rawData = s_data['rawData']
        contentData = s_data['content']
        wordData = s_data['word']

        # nFix: number of fixations
        # FFD: first fixation duration
        # TRT: total reading time
        # GD: gaze duration
        # GPT: go-past time
        gaze_features = ['nFix', 'FFD', 'TRT', 'GD', 'GPT']
        saccade_features = ['inSacc_velocity_mean', 'inSacc_duration_mean', 'outSacc_velocity_mean', 'outSacc_duration_mean', 'inSacc_amp_mean', 'outSacc_amp_mean', 'inSacc_velocity_max', 'inSacc_duration_max', 'outSacc_velocity_max', 'outSacc_duration_max', 'inSacc_amp_max', 'outSacc_amp_max']
        if config.saccades is True:
            gaze_features = gaze_features + saccade_features

        for idx in range(len(rawData)):

            obj_reference_content = contentData[idx][0]
            sent = dh.load_matlab_string(f[obj_reference_content])
            #print(sent)

            sent_features = {}
            # get word level data
            try:
                word_data = dh.extract_word_level_data(f, f[wordData[idx][0]],
                                                   eeg_float_resolution=dh.eeg_float_resolution)
                for widx in range(len(word_data)):
                    word_feats = []
                    for feature in gaze_features:
                        #print(feature)
                        #print(word_data[widx][feature])
                        if word_data[widx][feature] is not None:
                            word_feats.append(float(word_data[widx][feature]))
                        else:
                            word_feats.append(np.nan)
                    sent_features[widx] = word_feats
            except ValueError:
                print("NO sentence data available!")

            # save new features to dict (ignoring NR/TSR duplicates!)
            if sent_features and label == label_dict[sent]:
                if sent not in gaze_dict:
                    gaze_dict[sent] = {}
                    for widx, fts in sent_features.items():
                        gaze_dict[sent][widx] = [fts]
                else:
                    for widx, fts in sent_features.items():
                        gaze_dict[sent][widx].append(sent_features[widx])


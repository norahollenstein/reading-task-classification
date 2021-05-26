from . import data_loading_helpers as dh
import config
import numpy as np


def extract_word_raw_eeg(sentence_data, eeg_dict):
    """extract word-level raw EEG data of all sentences.
    word-level EEG data = mean activity over all fixations of a word"""

    for tup in sentence_data:

        # read Matlab v7.3 structures
        f = tup[0]
        s_data = tup[1]
        rawData = s_data['rawData']
        contentData = s_data['content']
        wordData = s_data['word']

        for idx in range(len(rawData)):

            obj_reference_content = contentData[idx][0]
            sent = dh.load_matlab_string(f[obj_reference_content])

            sent_features = {}
            # get word level data

            try:
                word_data = dh.extract_word_level_data(f, f[wordData[idx][0]],
                                               eeg_float_resolution=dh.eeg_float_resolution)

                for widx in range(len(word_data)):
                    word = word_data[widx]['content']
                    fixations_eeg = word_data[widx]["RAW_EEG"]

                    word_eeg = []
                    if len(fixations_eeg) > 0:
                        for fixation in fixations_eeg:
                            fix = np.nanmean(fixation, axis=0)
                            if not np.isnan(fix).any():
                                word_eeg.append(fix)
                        word_eeg = np.nanmean(word_eeg, axis=0)
                        if word_eeg.shape == (105,):
                            word_eeg = [float(n) for n in word_eeg]
                            sent_features[widx] = list(word_eeg[:104])

                    if widx not in sent_features:
                        nan_array = np.empty((104,))
                        nan_array[:] = np.NaN
                        nan_array = [float(n) for n in nan_array]
                        sent_features[widx] = nan_array

            except ValueError:
                print("NO sentence data available!")

            #if sent_features:
            # for sentiment and relation detection
            if config.class_task.startswith('read-task'):
                if sent not in eeg_dict:
                    eeg_dict[sent] = {}
                    for widx, fts in sent_features.items():
                        eeg_dict[sent][widx] = [fts]
                else:
                    for widx, fts in sent_features.items():
                        if not widx in eeg_dict[sent]:
                            eeg_dict[sent][widx] = [fts]
                        else:
                            eeg_dict[sent][widx].append(sent_features[widx])


def extract_word_band_eeg(sentence_data, eeg_dict, label_dict):
    """extract word-level raw EEG data of all sentences.
    word-level EEG data = mean activity over all fixations of a word"""

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
        band1, band2 = get_freq_band_data()

        for idx in range(len(rawData)):

            obj_reference_content = contentData[idx][0]
            sent = dh.load_matlab_string(f[obj_reference_content])

            sent_features = {}
            # get word level data

            try:
                word_data = dh.extract_word_level_data(f, f[wordData[idx][0]],
                                               eeg_float_resolution=dh.eeg_float_resolution)

                for widx in range(len(word_data)):
                    #word = word_data[widx]['content']
                    if word_data[widx]["RAW_EEG"]:

                        # t, a, b, or g
                        word_t1 = word_data[widx]["TRT_"+band1]
                        word_t2 = word_data[widx]["TRT_"+band2]
                        word_t = (word_t1 + word_t2) / 2
                        word_t = word_t.reshape(word_t.shape[0],)
                        word_t = [float(n) for n in word_t]
                        sent_features[widx] = word_t[:104]

                    else:
                        nan_array = np.empty((104,))
                        nan_array[:] = np.NaN
                        nan_array = [float(n) for n in nan_array]
                        sent_features[widx] = nan_array

            except ValueError:
                print("NO sentence data available!")

            # save new features to dict (ignoring NR/TSR duplicates!)
            if sent_features and label == label_dict[sent]:
                if sent not in eeg_dict:
                    eeg_dict[sent] = {}
                    for widx, fts in sent_features.items():
                        eeg_dict[sent][widx] = [fts]
                else:
                    for widx, fts in sent_features.items():
                        eeg_dict[sent][widx].append(sent_features[widx])


def extract_fix_band_eeg(sentence_data, eeg_dict):
    """extract fixation-level raw EEG data of all sentences:
    fixation-level EEG data = mean activity over the first fixation of a word"""

    for tup in sentence_data:

        # read Matlab v7.3 structures
        f = tup[0]
        s_data = tup[1]
        rawData = s_data['rawData']
        contentData = s_data['content']
        wordData = s_data['word']
        band1, band2 = get_freq_band_data()

        for idx in range(len(rawData)):

            obj_reference_content = contentData[idx][0]
            sent = dh.load_matlab_string(f[obj_reference_content])

            sent_features = {}
            # get word level data

            try:
                word_data = dh.extract_word_level_data(f, f[wordData[idx][0]],
                                               eeg_float_resolution=dh.eeg_float_resolution)

                for widx in range(len(word_data)):
                    #word = word_data[widx]['content']
                    if word_data[widx]["RAW_EEG"]:

                        # t, a, b, or g
                        word_t1 = word_data[widx]["FFD_"+band1]
                        word_t2 = word_data[widx]["FFD_"+band2]
                        #print(len(word_t1))
                        word_t = (word_t1 + word_t2) / 2
                        word_t = word_t.reshape(word_t.shape[0],)
                        word_t = [float(n) for n in word_t]
                        sent_features[widx] = word_t

                    else:
                        nan_array = np.empty((105,))
                        nan_array[:] = np.NaN
                        nan_array = [float(n) for n in nan_array]
                        sent_features[widx] = nan_array

            except ValueError:
                print("NO sentence data available!")

            #if sent_features:
            # for sentiment and relation detection
            if config.class_task.startswith('sentiment') or config.class_task == "reldetect":
                if sent not in eeg_dict:
                    eeg_dict[sent] = {}
                    for widx, fts in sent_features.items():
                        eeg_dict[sent][widx] = [fts]
                else:
                    for widx, fts in sent_features.items():
                        if not widx in eeg_dict[sent]:
                            eeg_dict[sent][widx] = [fts]
                        else:
                            eeg_dict[sent][widx].append(sent_features[widx])


def extract_sent_raw_eeg(sentence_data, eeg_dict):
    """extract sentence-level raw EEG data of all sentences."""

    for tup in sentence_data:

        # read Matlab v7.3 structures
        f = tup[0]
        s_data = tup[1]
        rawData = s_data['rawData']
        contentData = s_data['content']

        for idx in range(len(rawData)):

            raw_sent_eeg_ref = rawData[idx][0]
            raw_sent_eeg = f[raw_sent_eeg_ref]
            mean_raw_sent_eeg = np.nanmean(raw_sent_eeg, axis=0)
            #print(mean_raw_sent_eeg)

            obj_reference_content = contentData[idx][0]
            sent = dh.load_matlab_string(f[obj_reference_content])

            # for sentiment and relation detection
            if config.class_task.startswith('sentiment') or config.class_task == "reldetect":
                if sent not in eeg_dict:
                    eeg_dict[sent] = {'mean_raw_sent_eeg': [mean_raw_sent_eeg]}
                else:
                    eeg_dict[sent]['mean_raw_sent_eeg'].append(mean_raw_sent_eeg)

            # for ner (different tokenization needed for NER)
            #if config.class_task == "ner":
            # todo: how to handle for word-level models? of fixation level? or timestamp level?


def get_freq_band_data():

    if 'theta' in config.feature_set[0]:
        band1 = 't1'
        band2 = 't2'

    if 'alpha' in config.feature_set[0]:
        band1 = 'a1'
        band2 = 'a2'

    if 'beta' in config.feature_set[0]:
        band1 = 'b1'
        band2 = 'b2'

    if 'gamma' in config.feature_set[0]:
        band1 = 'g1'
        band2 = 'g2'

    return band1, band2


def extract_sent_freq_eeg(sentence_data, eeg_dict):
    """extract sentence-level frequency band EEG of all sentences."""

    for tup in sentence_data:

        # read Matlab v7.3 structures
        f = tup[0]
        s_data = tup[1]

        band1, band2 = get_freq_band_data()

        meanB1data = s_data['mean_'+band1]
        meanB2data = s_data['mean_'+band2]
        contentData = s_data['content']

        for idx in range(len(meanB1data)):

            sent_t1_ref = meanB1data[idx][0]
            sent_t1 = f[sent_t1_ref]

            sent_t2_ref = meanB2data[idx][0]
            sent_t2 = f[sent_t2_ref]

            mean_sent_t = (np.array(sent_t1) + np.array(sent_t2)) / 2.0
            mean_sent_t = mean_sent_t[:, 0]
            print(mean_sent_t.shape)

            obj_reference_content = contentData[idx][0]
            sent = dh.load_matlab_string(f[obj_reference_content])

            # for sentiment and relation detection
            if config.class_task.startswith('sentiment') or config.class_task == "reldetect":
                if sent not in eeg_dict:
                    eeg_dict[sent] = {config.feature_set[0]+'_sent_eeg': [mean_sent_t]}
                else:
                    eeg_dict[sent][config.feature_set[0]+'_sent_eeg'].append(mean_sent_t)

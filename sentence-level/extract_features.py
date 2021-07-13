import warnings
import data_loading_helpers as dlh
import numpy as np
import readability
from nltk import word_tokenize
import config


def flesch_reading_ease(text):
    """get Flesch reading ease score for a sentence."""
    #print(text)
    tokenized = " ".join(word_tokenize(text))
    #print(tokenized)

    results = readability.getmeasures(tokenized, lang='en')
    fre = results['readability grades']['FleschReadingEase']

    #print("Flesch:", fre)

    return fre


def relabel_sessions(idx, label_orig):
    if label_orig == "SR-Sess":
        print(idx, label_orig)
        if idx < 250:
            label = "Sess1"
        else:
            label = "Sess2"
    else:
        label = label_orig

    return label


def extract_sentence_features(subject, f, feature_set, feature_dict, label):
    """extract sentence level features from Matlab struct"""

    rawData = f['rawData']
    contentData = f['content']

    print(len(rawData))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        for idx, sent_data in enumerate(rawData):

            full_idx = len(feature_dict) + 1

            print(idx,full_idx, label)
            if config.class_task == "sessions":
                label = relabel_sessions(idx, label)

            obj_reference_content = contentData[idx][0]
            sent = dlh.load_matlab_string(f[obj_reference_content])

            # Flesch reading ease score
            fre = flesch_reading_ease(sent)

            # omission rate
            omissionR = f['omissionRate']
            obj_reference_omr = omissionR[idx][0]
            omr = np.array(f[obj_reference_omr])[0][0]

            # fixation number
            allFix = f['allFixations']
            obj_reference_allFix = allFix[idx][0]
            af = f[obj_reference_allFix]

            # mean saccade amplitude
            saccMeanAmp = f['saccMeanAmp']
            obj_reference_saccMeanAmp = saccMeanAmp[idx][0]
            smeana = np.array(f[obj_reference_saccMeanAmp])[0][0]
            smeana = smeana if not np.isnan(smeana) else 0.0


            # mean saccade duration
            saccMeanDur = f['saccMeanDur']
            obj_reference_saccMeanDur = saccMeanDur[idx][0]
            smeand = np.array(f[obj_reference_saccMeanDur])[0][0]
            smeand = smeand if not np.isnan(smeand) else 0.0

            # mean saccade velocity
            saccMeanVel = f['saccMeanVel']
            obj_reference_saccMeanVel = saccMeanVel[idx][0]
            smeanv = np.array(f[obj_reference_saccMeanVel])[0][0]
            smeanv = smeanv if not np.isnan(smeanv) else 0.0

            # saccade max amplitude
            saccMaxAmp = f['saccMaxAmp']
            obj_reference_saccMaxAmp = saccMaxAmp[idx][0]
            try:
                smaxa = np.array(f[obj_reference_saccMaxAmp])[0][0]
                smaxa = smaxa if not np.isnan(smaxa) else 0.0
            except IndexError:
                smaxa = 0.0

            # saccade max velocity
            saccMaxVel = f['saccMaxVel']
            obj_reference_saccMaxVel = saccMaxVel[idx][0]
            try:
                smaxv = np.array(f[obj_reference_saccMaxVel])[0][0]
                smaxv = smaxv if not np.isnan(smaxv) else 0.0
            except IndexError:
                smaxv = 0.0

            # saccade max duration
            saccMaxDur = f['saccMaxDur']
            obj_reference_saccMaxDur = saccMaxDur[idx][0]
            try:
                smaxd = np.array(f[obj_reference_saccMaxDur])[0][0]
                smaxd = smaxd if not np.isnan(smaxd) else 0.0
            except IndexError:
                smaxd = 0.0

            # EEG means
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                theta1 = f['mean_t1']
                obj_reference_t1 = theta1[idx][0]
                theta2 = f['mean_t2']
                obj_reference_t2 = theta2[idx][0]
                t_electrodes = np.nanmean(np.array([np.array(f[obj_reference_t1])[:104],np.array(f[obj_reference_t2])[:104]]), axis=0)
                t_mean = np.nanmean(t_electrodes)

                alpha1 = f['mean_a1']
                obj_reference_a1 = alpha1[idx][0]
                alpha2 = f['mean_a2']
                obj_reference_a2 = alpha2[idx][0]
                a_electrodes = np.nanmean(
                    np.array([np.array(f[obj_reference_a1])[:104], np.array(f[obj_reference_a2])[:104]]), axis=0)
                a_mean = np.nanmean(a_electrodes)

                beta1 = f['mean_b1']
                obj_reference_b1 = beta1[idx][0]
                beta2 = f['mean_b2']
                obj_reference_b2 = beta2[idx][0]
                b_electrodes = np.nanmean(
                    np.array([np.array(f[obj_reference_b1])[:104], np.array(f[obj_reference_b2])[:104]]), axis=0)
                b_mean = np.nanmean(b_electrodes)

                gamma1 = f['mean_g1']
                obj_reference_g1 = gamma1[idx][0]
                gamma2 = f['mean_g2']
                obj_reference_g2 = gamma2[idx][0]
                g_electrodes = np.nanmean(np.array([np.array(f[obj_reference_g1])[:104],np.array(f[obj_reference_g2])[:104]]), axis=0)
                g_mean = np.nanmean(g_electrodes)

            ### --- Text difficulty baseline --- ###
            if feature_set == "flesch_baseline":
                feature_dict[feature_set][subject + "_" + label + "_" + str(idx)] = [fre, label]

            ### --- Sentencel-level eye tracking features --- ###
            if feature_set == "omission_rate":
                if not np.isnan(omr).any():
                    feature_dict[feature_set][subject + "_" + label + "_" + str(idx)] = [omr, label]

            elif feature_set == "fixation_number":
                if 'duration' in af:
                    weighted_nFix = np.array(af['duration']).shape[0] / len(sent.split())
                    feature_dict[feature_set][subject + "_" + label + "_" + str(idx)] = [weighted_nFix, label]

            elif feature_set == "reading_speed":
                if 'duration' in af:
                    # convert sample to seconds
                    weighted_speed = (np.sum(np.array(af['duration']))*2/100) / len(sent.split())
                    feature_dict[feature_set][subject + "_" + label + "_" + str(idx)] = [weighted_speed, label]

            elif feature_set == "mean_sacc_dur":
                    feature_dict[feature_set][subject + "_" + label + "_" + str(idx)] = [smeand, label]

            elif feature_set == "mean_sacc_amp":
                    feature_dict[feature_set][subject + "_" + label + "_" + str(idx)] = [smeana, label]

            elif feature_set == "max_sacc_velocity":
                    feature_dict[feature_set][subject + "_" + label + "_" + str(idx)] = [smaxv, label]

            elif feature_set == "mean_sacc_velocity":
                    feature_dict[feature_set][subject + "_" + label + "_" + str(idx)] = [smeanv, label]

            elif feature_set == "max_sacc_dur":
                    feature_dict[feature_set][subject + "_" + label + "_" + str(idx)] = [smaxd, label]

            elif feature_set == "max_sacc_amp":
                    feature_dict[feature_set][subject + "_" + label + "_" + str(idx)] = [smaxa, label]

            elif feature_set == "sent_gaze":
                if 'duration' in af:
                    weighted_nFix = np.array(af['duration']).shape[0] / len(sent.split())
                    weighted_speed = (np.sum(np.array(af['duration'])) * 2 / 100) / len(sent.split())
                    feature_dict[feature_set][subject + "_" + label + "_" + str(idx)] = [omr, weighted_nFix, weighted_speed, smeand, label]

            elif feature_set == "sent_gaze_sacc":
                if 'duration' in af:
                    weighted_nFix = np.array(af['duration']).shape[0] / len(sent.split())
                    weighted_speed = (np.sum(np.array(af['duration'])) * 2 / 100) / len(sent.split())
                    feature_dict[feature_set][subject + "_" + label + "_" + str(idx)] = [omr, weighted_nFix, weighted_speed, smeand, smaxv, smeanv, smaxd, smeana, smaxa, label]

            elif feature_set == "sent_saccade":
                if 'duration' in af:
                    feature_dict[feature_set][subject + "_" + label + "_" + str(idx)] = [smeand, smaxv, smeanv, smaxd, smeana, smaxa, label]

            ### --- Sentencel-level EEG features --- ###
            elif feature_set == "theta_mean":
                if not np.isnan(t_mean):
                    feature_dict[feature_set][subject + "_" + label + "_" + str(idx)] = [t_mean, label]

            elif feature_set == "alpha_mean":
                if not np.isnan(a_mean):
                    feature_dict[feature_set][subject + "_" + label + "_" + str(idx)] = [a_mean, label]

            elif feature_set == "beta_mean":
                if not np.isnan(b_mean):
                    feature_dict[feature_set][subject + "_" + label + "_" + str(idx)] = [b_mean, label]

            elif feature_set == "gamma_mean":
                if not np.isnan(g_mean):
                    feature_dict[feature_set][subject + "_" + label + "_" + str(idx)] = [g_mean, label]

            elif feature_set == "eeg_means":
                if not np.isnan(g_mean) and not np.isnan(t_mean) and not np.isnan(b_mean) and not np.isnan(a_mean):
                    feature_dict[feature_set][subject + "_" + label + "_" + str(idx)] = [t_mean, a_mean, b_mean, g_mean, label]

            elif feature_set == "electrode_features_theta":
                if not np.isnan(t_electrodes).any():
                    feature_dict[feature_set][subject + "_" + label + "_" + str(idx)] = np.concatenate(t_electrodes).ravel().tolist() + [label]

            elif feature_set == "electrode_features_alpha":
                if not np.isnan(a_electrodes).any():
                    feature_dict[feature_set][subject + "_" + label + "_" + str(idx)] = np.concatenate(a_electrodes).ravel().tolist() + [label]

            elif feature_set == "electrode_features_beta":
                if not np.isnan(b_electrodes).any():
                    feature_dict[feature_set][subject + "_" + label + "_" + str(idx)] = np.concatenate(b_electrodes).ravel().tolist() + [label]

            elif feature_set == "electrode_features_gamma":
                if not np.isnan(g_electrodes).any():
                    feature_dict[feature_set][subject + "_" + label + "_" + str(idx)] = np.concatenate(g_electrodes).ravel().tolist() + [label]

            elif feature_set == "electrode_features_all":
                if not np.isnan(g_electrodes).any() and not np.isnan(a_electrodes).any() and not np.isnan(t_electrodes).any() and not np.isnan(b_electrodes).any():
                    feature_dict[feature_set][subject + "_" + label + "_" + str(idx)] = np.concatenate(t_electrodes).ravel().tolist() + np.concatenate(a_electrodes).ravel().tolist() + np.concatenate(b_electrodes).ravel().tolist() + np.concatenate(g_electrodes).ravel().tolist() + [label]

            elif feature_set == "sent_gaze_eeg_means":
                if 'duration' in af and not np.isnan(g_mean) and not np.isnan(t_mean) and not np.isnan(b_mean) and not np.isnan(a_mean):
                    weighted_nFix = np.array(af['duration']).shape[0] / len(sent.split())
                    weighted_speed = (np.sum(np.array(af['duration'])) * 2 / 100) / len(sent.split())
                    feature_dict[feature_set][subject + "_" + label + "_" + str(idx)] = [omr, weighted_nFix, weighted_speed,
                                                                                         smeand, smaxv, smeanv, smaxd, t_mean, a_mean, b_mean, g_mean,
                                                                                         label]

            else:
                print(feature_set, "IS NOT A VALID FEATURE SET.")

        return feature_dict


def extract_fixation_features(subject, f, feature_set, feature_dict, label):
    """parse Matlab struct to extract EEG singals only for fixation occurring inside wordbounds"""
    """ extract features in the order they were read"""
    rawData = f['rawData']
    contentData = f['content']

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for idx in range(len(rawData)):
            # get word level data
            wordData = f['word']

            try:
                word_data = dlh.extract_word_level_data(f, f[wordData[idx][0]])

                fix_order_raw_eeg = []
                fixations_indices = {}
                # get EEG data in order of fixations
                for widx in word_data["word_reading_order"]:
                    if len(word_data[widx]["RAW_EEG"]) > 1:
                        if widx not in fixations_indices:
                            fixations_indices[widx] = 0
                        else:
                            fixations_indices[widx] += 1
                        fixation_avg = np.nanmean(word_data[widx]["RAW_EEG"][fixations_indices[widx]], axis=0)
                        #print(fixation_avg)
                    else:
                        fixation_avg = np.nanmean(word_data[widx]["RAW_EEG"][0], axis=0)
                    fix_order_raw_eeg.append(fixation_avg)

                #print(fix_order_raw_eeg[0])

                """
                word_g1_electrodes = []; word_g2_electrodes = [];
                sent_feats = []; sent_trt_t1 = []; sent_trt_t2 = []; sent_trt_a1 = []; sent_trt_a2 = []; sent_trt_b1 = [];
                sent_trt_b2 = []; sent_trt_g1 = []; sent_trt_g2 = [];
                # get EEG data in order of words
                for widx in range(len(word_data)-1):
                    if word_data[widx]["RAW_EEG"]:
                        # "fix_avg_raw_eeg"
                        fixation_avg = [np.mean(fix) for fix in word_data[widx]["RAW_EEG"]]
                        sent_feats.append(np.nanmean(fixation_avg))
    
                        # todo: try with ICA features
    
                        # "fix_eeg_means"
                        word_t1 = np.mean(word_data[widx]["TRT_t1"])
                        sent_trt_t1.append(word_t1)
                        word_t2 = np.mean(word_data[widx]["TRT_t2"])
                        sent_trt_t2.append(word_t2)
    
                        word_a1 = np.mean(word_data[widx]["TRT_a1"])
                        sent_trt_a1.append(word_a1)
                        word_a2 = np.mean(word_data[widx]["TRT_a2"])
                        sent_trt_a2.append(word_a2)
    
                        word_b1 = np.mean(word_data[widx]["TRT_b1"])
                        sent_trt_b1.append(word_b1)
                        word_b2 = np.mean(word_data[widx]["TRT_b2"])
                        sent_trt_b2.append(word_b2)
    
                        word_g1 = np.mean(word_data[widx]["TRT_g1"])
                        sent_trt_g1.append(word_g1)
                        word_g2 = np.mean(word_data[widx]["TRT_g2"])
                        sent_trt_g2.append(word_g2)
    
                        # "fix_electrode_features_gamma"
                        word_g1_electrodes.append(word_data[widx]["TRT_g1"])
                        word_g2_electrodes.append(word_data[widx]["TRT_g2"])
    
                if feature_set == "fix_avg_raw_eeg" and sent_feats:
                    if not np.isnan(sent_feats).any():
                        feature_dict[feature_set][subject + "_" + label_orig + "_" + str(idx)] = [np.nanmean(sent_feats), label_orig]
    
                elif feature_set == "fix_eeg_means" and sent_trt_t1:
                    if not np.isnan(sent_trt_t1).any():
                        feature_dict[feature_set][subject + "_" + label_orig + "_" + str(idx)] = [np.nanmean(sent_trt_t1), np.nanmean(sent_trt_t2), np.nanmean(sent_trt_a1), np.nanmean(sent_trt_a2), np.nanmean(sent_trt_b1), np.nanmean(sent_trt_b2), np.nanmean(sent_trt_g1), np.nanmean(sent_trt_g2), label_orig]
    
                elif feature_set == "fix_gamma_means" and sent_trt_g1:
                    if not np.isnan(sent_trt_g1).any():
                        feature_dict[feature_set][subject + "_" + label_orig + "_" + str(idx)] = [np.nanmean(sent_trt_g1), np.nanmean(sent_trt_g2), label_orig]
    
                elif feature_set == "fix_electrode_features_gamma" and word_g1_electrodes:
                    if not np.isnan(word_g1_electrodes).any():
                        feat_list = np.hstack((np.nanmean(word_g1_electrodes, axis=0), np.nanmean(word_g2_electrodes, axis=0)))
                        feature_dict[feature_set][subject + "_" + label_orig + "_" + str(idx)] = list(np.mean(feat_list, axis=1)) + [label_orig]
    
                elif feature_set == "fix_electrode_features_gamma_10%" and word_g1_electrodes:
                    if not np.isnan(word_g1_electrodes).any():
                        #print(len(word_g1_electrodes), len(word_g2_electrodes))
                        # take 10%, but at least 1 word
                        p10 = max(round(len(word_g1_electrodes)/10), 1)
                        feat_list = np.hstack((np.nanmean(word_g1_electrodes[:p10], axis=0), np.nanmean(word_g2_electrodes[:p10], axis=0)))
                        feature_dict[feature_set][subject + "_" + label_orig + "_" + str(idx)] = list(np.mean(feat_list, axis=1)) + [label_orig]
    
                elif feature_set == "fix_electrode_features_gamma_20%" and word_g1_electrodes:
                    if not np.isnan(word_g1_electrodes).any():
                        # take 10%, but at least 1 word
                        p20 = max(round(len(word_g1_electrodes)/5), 1)
                        feat_list = np.hstack((np.nanmean(word_g1_electrodes[:p20], axis=0), np.nanmean(word_g2_electrodes[:p20], axis=0)))
                        feature_dict[feature_set][subject + "_" + label_orig + "_" + str(idx)] = list(np.mean(feat_list, axis=1)) + [label_orig]
    
                """

                if feature_set == 'fix_order_raw_eeg_electrodes' and fix_order_raw_eeg:
                    avg = np.nanmean(fix_order_raw_eeg, axis=0)
                    if not np.isnan(avg).any():
                        feature_dict[feature_set][subject + "_" + label + "_" + str(idx)] = list(avg) + [label]

                elif feature_set == 'fix_order_raw_eeg_electrodes_10%' and fix_order_raw_eeg:
                    p10 = max(round(len(fix_order_raw_eeg) / 10), 1) # at least 1 fixation if sentence contains <10
                    avg = np.nanmean(fix_order_raw_eeg[:p10], axis=0)
                    #print(avg)
                    if not np.isnan(avg).any():
                        feature_dict[feature_set][subject + "_" + label + "_" + str(idx)] = list(avg) + [label]

                elif feature_set == 'fix_order_raw_eeg_electrodes_20%' and fix_order_raw_eeg:
                    p20 = max(round(len(fix_order_raw_eeg) / 5), 1)
                    avg = np.nanmean(fix_order_raw_eeg[:p20], axis=0)
                    if not np.isnan(avg).any():
                        feature_dict[feature_set][subject + "_" + label + "_" + str(idx)] = list(avg) + [label]

                elif feature_set == 'fix_order_raw_eeg_electrodes_50%' and fix_order_raw_eeg:
                    p50 = max(round(len(fix_order_raw_eeg) / 2), 1)
                    avg = np.nanmean(fix_order_raw_eeg[:p50], axis=0)
                    if not np.isnan(avg).any():
                        feature_dict[feature_set][subject + "_" + label + "_" + str(idx)] = list(avg) + [label]

                elif feature_set == 'fix_order_raw_eeg_electrodes_75%' and fix_order_raw_eeg:
                    p75 = max(round((len(fix_order_raw_eeg) / 10)*7.5), 1)
                    avg = np.nanmean(fix_order_raw_eeg[:p75], axis=0)
                    if not np.isnan(avg).any():
                        feature_dict[feature_set][subject + "_" + label + "_" + str(idx)] = list(avg) + [label]

            except ValueError:
                print("NO WORD DATA AVAILABLE for sentence ", idx)

    return feature_dict

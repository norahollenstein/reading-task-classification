
# dataset {zuco1, zuco2, zucoAll, zuco1sr}
dataset = 'zuco1'

if dataset is 'zuco2':
    # todo: what about missing subjects?
    subjects = ['YAC', 'YAG', 'YAK', 'YDG', 'YDR', 'YFR', 'YFS', 'YHS', 'YIS', 'YLS', 'YMD', 'YRK', 'YRP', 'YSD', 'YSL', 'YTL']  # exclude YMH,  - YRH, YMS
    # new preprocessed data Sept. 2020
    rootdir = "/mnt/ds3lab-scratch/noraho/datasets/zuco/zuco2_preprocessed_sep2020/"
elif dataset is 'zuco1' or dataset is 'zuco1sr':
    subjects = ["ZDN", "ZPH", "ZJN", "ZAB", "ZJM", "ZKB", "ZKH", "ZMG", "ZGW", "ZKW", "ZDM"] #"ZJS"
    rootdir = "/mnt/ds3lab-scratch/noraho/datasets/zuco/zuco1_preprocessed_sep2020/"
elif dataset is "zucoAll":
    subjects = ['YAC', 'YAG', 'YAK', 'YDG', 'YDR', 'YFR', 'YFS', 'YHS', 'YIS', 'YLS', 'YMD', 'YRK', 'YRP', 'YSD', 'YSL', 'YTL', "ZDN", "ZPH", "ZJN", "ZAB", "ZJM", "ZKB", "ZKH", "ZMG", "ZGW", "ZKW", "ZDM"] #'YMS', 'YRH', #'ZJS
    rootdir2 = "/mnt/ds3lab-scratch/noraho/datasets/zuco/zuco2_preprocessed_sep2020/"
    rootdir1 = "/mnt/ds3lab-scratch/noraho/datasets/zuco/zuco1_preprocessed_sep2020/"


# level {word, sentence}
# sentence-level: SVM used for classification
# word-level: Only data from fixations inside wordbounds
level = 'sentence'

# experiment setup
seed = 1
runs = 50

# randomize labels as a sanity check; default = False
randomized_labels = False

if level is 'sentence':
    # sentence-level eye-tracking feature sets
    #feature_sets = ["fixation_number", "omission_rate", "reading_speed", 'sent_gaze', "saccade_duration", "saccade_max_velocity", 'sent_saccade', 'sent_gaze_sacc']
    feature_sets = ['sent_saccade']
    # sentence-level EEG feature sets
    # "eeg_means", "eeg_diffs", "eeg_means_diffs", "sent_gaze_eeg_means"]
    #feature_sets = ["eeg_means", "eeg_diffs", "eeg_means_diffs", "sent_gaze_eeg_means", "electrode_features_theta", "electrode_features_alpha", "electrode_features_beta", "electrode_features_gamma", "electrode_features_all", "electrode_features_gamma_1", "electrode_features_gamma_2", "gamma_means", "gamma_diffs", "gamma_means_diffs"]
    #feature_sets = ["omission_rate", "fixation_number", "weighted_fixation_number", "reading_speed", "weighted_reading_speed", "sent_gaze", "eeg_means", "eeg_diffs", "eeg_means_diffs", "sent_gaze_eeg_means", "electrode_features_theta", "electrode_features_alpha", "electrode_features_beta", "electrode_features_gamma", "electrode_features_all", "electrode_features_gamma_1", "electrode_features_gamma_2", "gamma_means", "gamma_diffs", "gamma_means_diffs"]
    # gamma feature sets: "electrode_features_gamma_1", "electrode_features_gamma_2", "gamma_means", "gamma_diffs", "gamma_means_diffs"
    #feature_sets = ["electrode_features_gamma_1", "electrode_features_gamma_2", "gamma_means", "gamma_diffs", "gamma_means_diffs"]

elif level is 'word':
    # word-level
    #feature_sets = ['fix_avg_raw_eeg', 'fix_eeg_means', 'fix_gamma_means', 'fix_electrode_features_gamma']
    feature_sets = ['fix_electrode_features_gamma_10%', 'fix_electrode_features_gamma_20%', 'fix_electrode_features_gamma']
    #feature_sets = ['fix_order_raw_eeg_electrodes_10%', 'fix_order_raw_eeg_electrodes_20%', 'fix_order_raw_eeg_electrodes_50%', 'fix_order_raw_eeg_electrodes_75%', 'fix_order_raw_eeg_electrodes']

# classification task {tasks, sessions, subjects}
class_task = 'tasks'
if class_task == 'tasks' or class_task == 'tasks-cross-subj':
    target_names = ["TSR", "NR"]
    labels = [0, 1]
elif class_task == 'sessions':
    target_names = ["Sess1", "Sess2"]
    labels = [0, 1]
elif class_task == "subjects":
    target_names = subjects
    labels = list(range(len(subjects)))

# SVM params
kernel = 'linear' # only linear kernel allows for analysis of coefficients
train_split = 0.9 # 90% of the data used for training
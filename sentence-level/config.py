
# dataset {zuco1, zuco2, zucoAll, zuco1sr, zuco1sr_only}
dataset = 'zuco2'

if dataset is 'zuco2':
    subjects = ['YAK']#, 'YMD', 'YTL', 'YRP', 'YDR', 'YHS']
    #subjects = ['YAC', 'YAG', 'YAK']#, 'YDG', 'YDR', 'YFR', 'YFS', 'YHS', 'YIS', 'YLS', 'YMD', 'YRK', 'YRP', 'YSD', 'YSL', 'YTL']  # exclude YMH,  - YRH, YMS
    # new preprocessed data Sept. 2020
    rootdir = "/mnt/ds3lab-scratch/noraho/datasets/zuco/zuco2_preprocessed_sep2020/"
    rootdir = "../../mehtlab_loc/ETH_AS/FirstLevelV2_concat_unfold_correctlyMergedSacc_avgref/"
elif dataset is 'zuco1' or dataset is 'zuco1sr':
    subjects = ["ZDN", "ZPH", "ZJN", "ZAB", "ZJM", "ZKB", "ZKH", "ZMG", "ZGW", "ZKW", "ZDM"] #"ZJS"
    rootdir = "/mnt/ds3lab-scratch/noraho/datasets/zuco/zuco1_preprocessed_sep2020/"
    rootdir_sr = "/mnt/ds3lab-scratch/noraho/datasets/zuco/zuco1_SR_preprocessed_apr2021/"
elif dataset is 'zuco1sr_only':
    subjects = ["ZDN", "ZPH", "ZJN", "ZAB", "ZJM", "ZKB", "ZKH", "ZMG", "ZGW", "ZKW", "ZDM"]  # "ZJS"
    rootdir_sr = "/mnt/ds3lab-scratch/noraho/datasets/zuco/zuco1_SR_preprocessed_apr2021/"
elif dataset is "zucoAll":
    subjects = ['YAC', 'YAG', 'YAK', 'YDG', 'YDR', 'YFR', 'YFS', 'YHS', 'YIS', 'YLS', 'YMD', 'YRK', 'YRP', 'YSD', 'YSL', 'YTL', "ZDN", "ZPH", "ZJN", "ZAB", "ZJM", "ZKB", "ZKH", "ZMG", "ZGW", "ZKW", "ZDM"] #'YMS', 'YRH', #'ZJS
    subjects = ["ZDN", "ZPH", "ZJN", "ZAB", "ZJM", "ZKB", "ZKH", "ZMG", "ZGW", "ZKW", "ZDM"] #"ZJS"
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
    feature_sets = ["fixation_number", "omission_rate", "reading_speed", 'sent_gaze', "mean_sacc_dur", "max_sacc_velocity", "mean_sacc_velocity", "max_sacc_dur", "max_sacc_amp", "mean_sacc_amp", 'sent_saccade', 'sent_gaze_sacc', "theta_mean", "alpha_mean", "beta_mean", "gamma_mean", "eeg_means", "sent_gaze_eeg_means",  "electrode_features_theta", "electrode_features_alpha", "electrode_features_beta", "electrode_features_gamma", "electrode_features_all"]
    #feature_sets = ["max_sacc_amp", "mean_sacc_amp", 'sent_saccade', 'sent_gaze_sacc', "sent_gaze_eeg_means"]
    # sentence-level EEG feature sets
    #feature_sets = ["theta_mean", "alpha_mean", "beta_mean", "gamma_mean", "eeg_means", "sent_gaze_eeg_means", "electrode_features_theta", "electrode_features_alpha", "electrode_features_beta", "electrode_features_gamma", "electrode_features_all"]
    feature_sets = ["electrode_features_gamma"]
    # sentence-level baseline feature
    #feature_sets = ["flesch_baseline"]

elif level is 'word':
    # word-level
    #feature_sets = ['fix_avg_raw_eeg', 'fix_eeg_means', 'fix_gamma_means', 'fix_electrode_features_gamma']
    #feature_sets = ['fix_electrode_features_gamma_10%']#, 'fix_electrode_features_gamma_20%', 'fix_electrode_features_gamma']
    feature_sets = ['fix_order_raw_eeg_electrodes_10%']#, 'fix_order_raw_eeg_electrodes_20%', 'fix_order_raw_eeg_electrodes_50%', 'fix_order_raw_eeg_electrodes_75%', 'fix_order_raw_eeg_electrodes']
    feature_sets = ['fix_electrode_features_gamma_10%', 'fix_electrode_features_gamma_20%', 'fix_electrode_features_gamma_50%', 'fix_electrode_features_gamma_75%', 'fix_electrode_features_gamma']

# classification task {tasks, sessions, subjects, tasks-cross-subj, blocks, blocks-in-sets}
class_task = 'tasks_blocks'

if class_task == 'blocks-in-sets' or "tasks_blocks":
    set_in_train = 6

# SVM params
kernel = 'linear' # only linear kernel allows for analysis of coefficients
train_split = 0.9 # 90% of the data used for training


# EEG information
chanlocs = ['E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E9', 'E10', 'E11', 'E12', 'E13', 'E15', 'E16', 'E18', 'E19', 'E20',
            'E22',
            'E23', 'E24', 'E26', 'E27', 'E28', 'E29', 'E30', 'E31', 'E33', 'E34', 'E35', 'E36', 'E37', 'E38', 'E39',
            'E40',
            'E41', 'E42', 'E43', 'E44', 'E45', 'E46', 'E47', 'E50', 'E51', 'E52', 'E53', 'E54', 'E55', 'E57', 'E58',
            'E59',
            'E60', 'E61', 'E62', 'E64', 'E65', 'E66', 'E67', 'E69', 'E70', 'E71', 'E72', 'E74', 'E75', 'E76', 'E77',
            'E78',
            'E79', 'E80', 'E82', 'E83', 'E84', 'E85', 'E86', 'E87', 'E89', 'E90', 'E91', 'E92', 'E93', 'E95', 'E96',
            'E97',
            'E98', 'E100', 'E101', 'E102', 'E103', 'E104', 'E105', 'E106', 'E108', 'E109', 'E110', 'E111', 'E112',
            'E114',
            'E115', 'E116', 'E117', 'E118', 'E120', 'E121', 'E122', 'E123', 'E124']


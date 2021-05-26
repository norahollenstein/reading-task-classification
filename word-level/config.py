# dataset directories
#rootdir_zuco1 = "/Volumes/methlab/NLP/Ce_ETH/OSF-ZuCo1.0-200107/mat7.3/"
#rootdir_zuco2 = "/Volumes/methlab/NLP/Ce_ETH/2019/FirstLevel_V2/"
#base_dir = "/mnt/ds3lab-scratch/noraho/coling2020/"
#rootdir_zuco1 = base_dir+"zuco1/"
#rootdir_zuco2 = base_dir+"zuco2/"
rootdir_zuco1 = "/mnt/ds3lab-scratch/noraho/datasets/zuco/zuco1_preprocessed_sep2020/"
rootdir_zuco2 = "/mnt/ds3lab-scratch/noraho/datasets/zuco/zuco2_preprocessed_sep2020/"

# subjects (subejcts starting with "Z" are from ZuCo 1, subjects starting with "Y" are from ZuCo 2)
# note: for running the experiments with previously extracted and saved features only one subject (from each dataset) is necessary
subjects = ['YAG', 'YAC', 'YAK', 'YDG', 'YDR', 'YFR', 'YFS', 'YHS', 'YIS', 'YLS', 'YMD', 'YMS', 'YRH', 'YRK', 'YRP', 'YSD', 'YSL', 'YTL']  # exclude YMH
#subjects = ["ZKW"]#], "ZJS"]#, "ZDN"]#, "ZJN"]#, "ZPH", "ZAB", "ZJM", "ZKB", "ZKH", "ZMG", "ZGW", "ZKW", "ZDM"]


run_feature_extraction = True

# ML task {read-task, session, subjects}
class_task = 'read-task'

# features sets {'text_only' , 'eeg_raw', 'eeg_theta', 'eeg_alpha', 'eeg_beta', 'eeg_gamma', 'combi_eeg_raw', 'eye_tracking', 'combi_eye_tracking'}
# sentence level features: {'combi_concat', 'sent_eeg_theta'}
# combined models: {'eeg_eye_tracking', 'eeg4'}

saccades = True

feature_set = ['eye_tracking']

# word embeddings {none, glove (300d), bert}
embeddings = 'none'

# hyper-parameters to test - general
lstm_dim = [64]
lstm_layers = [1]
dense_dim = [64]
dropout = [0.1]
batch_size = [40]
epochs = [200]
lr = [0.001]


# other parameters
folds = 3
random_seed_values = [13]#, 78, 22, 66, 42]
validation_split = 0.1
patience = 10
min_delta = 0.0000001

data_percentage = 0


from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import random
import config
import sys
import mne
from mne import EvokedArray
from mne.decoding import Vectorizer, get_coef
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

# import a linear classifier from mne.decoding
from mne.decoding import LinearModel


def decode_svm_cooefficients(X, y, subj):
    """Source: https://mne.tools/stable/auto_examples/decoding/linear_model_patterns.html"""

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

    info = mne.create_info(ch_names=chanlocs, ch_types="eeg", sfreq=500)

    #epochs = mne.EvokedArray(data=np.transpose(train_X), info=info)
    clf = make_pipeline(
        Vectorizer(),  # 1) vectorize across time and channels
        StandardScaler(),  # 2) normalize features across trials
        LinearModel(
            LogisticRegression(solver='lbfgs')))
    clf.fit(X, y)

    # Extract and plot patterns and filters
    for name in ('patterns_', 'filters_'):
        # The `inverse_transform` parameter will call this method on any estimator
        # contained in the pipeline, in reverse order.
        coef = get_coef(clf, name, inverse_transform=True)

        evoked = EvokedArray(coef.reshape(-1,1), info=info)
        evoked.set_montage("GSN-HydroCel-128")

        fig, ax = plt.subplots(figsize=(7.5, 4.5), nrows=1, ncols=1)
        ax = evoked.plot_topomap(title='EEG %s' % name[:-1], time_unit='s')
        plt.savefig("test-topo-"+name[:-1]+"-"+subj+".pdf")


def svm(samples, seed_value, run, randomized=False):
    X = []
    y = []

    if config.class_task == "tasks":
        for sample_id, features in samples.items():
            subj = sample_id.split('_')[0]
            X.append(features[:-1])
            if randomized is False:
                if features[-1] == "NR":
                    y.append(1)
                else:
                    y.append(0)
            else:
                y.append(random.choice([0, 1]))

    elif config.class_task == "subjects":
        for sample_id, features in samples.items():
            X.append(features[:-1])
            if randomized is False:
                subj = sample_id.split('_')[0]
                subject_index = config.subjects.index(subj)
                y.append(subject_index)
            else:
                y.append(random.choice(list(range(len(config.subjects)))))

    elif config.class_task == "sessions":
        for sample_id, features in samples.items():

            if randomized is False:
                if features[-1] == "Sess1" and y.count(1) < 150:
                    X.append(features[:-1])
                    y.append(1)
                elif features[-1] == "Sess2" and y.count(0) < 150:
                    X.append(features[:-1])
                    y.append(0)

            else:
                y.append(random.choice([0, 1]))

        print("Samples:")
        print(y.count(0))
        print(y.count(1))

    elif config.class_task == "blocks":
        for sample_id, features in samples.items():
            X.append(features[:-1])
            blocks = ["NR_block1", "TSR_block1", "NR_block2", "TSR_block2", "NR_block3", "TSR_block3", "NR_block4",
                      "TSR_block4", "NR_block5", "TSR_block5", "NR_block6", "TSR_block6", "NR_block7", "TSR_block7"]

            if randomized is False:
                block = sample_id.split('_')[1] + "_" + sample_id.split('_')[2]

                block_index = blocks.index(block)
                y.append(block_index)
            else:
                y.append(random.choice(list(range(len(blocks)))))

    else:
        sys.exit("Classification task {0} not defined!".format(config.class_task))

    np.random.seed(seed_value)
    shuffled_X, shuffled_y = shuffle(X, y)

    # split into train/test
    size = int(config.train_split * len(shuffled_X))
    train_X = shuffled_X[:size]
    test_X = shuffled_X[size:]
    train_y = shuffled_y[:size]
    test_y = shuffled_y[size:]

    # scale feature values
    scaling = MinMaxScaler(feature_range=(-1, 1)).fit(train_X)
    train_X = scaling.transform(train_X)
    test_X = scaling.transform(test_X)

    # train SVM classifier
    clf = SVC(random_state=seed_value, kernel=config.kernel, gamma='scale', cache_size=1000)
    clf.fit(train_X, train_y)
    predictions = clf.predict(test_X)
    accuracy = len([i for i, j in zip(predictions, test_y) if i == j]) / len(test_y)



    # get coefficients
    if config.kernel is 'linear':
        coefficients = clf.coef_
    else:
        coefficients = [[]]

    if run == 49:
        decode_svm_cooefficients(train_X, train_y, subj)

    return predictions, test_y, accuracy, coefficients


def svm_cross_subj(samples, seed_value, test_subject, randomized=False):
    train_X = []
    train_y = []
    test_X = []
    test_y = []

    if config.class_task == "tasks-cross-subj":

        for sample_id, features in samples.items():

            if sample_id[:3] != test_subject:

                train_X.append(features[:-1])
                if randomized is False:
                    if features[-1] == "NR":
                        train_y.append(1)
                    else:
                        train_y.append(0)
                else:
                    train_y.append(random.choice([0, 1]))
            else:

                test_X.append(features[:-1])
                if randomized is False:
                    if features[-1] == "NR":
                        test_y.append(1)
                    else:
                        test_y.append(0)
                else:
                    test_y.append(random.choice([0, 1]))

    elif config.class_task == "blocks-in-sets":

        blocks = ["NR_block1", "TSR_block1", "NR_block2", "TSR_block2", "NR_block3", "TSR_block3", "NR_block4",
                  "TSR_block4", "NR_block5", "TSR_block5", "NR_block6", "TSR_block6", "NR_block7", "TSR_block7"]

        for sample_id, features in samples.items():

            block = sample_id.split('_')[1] + "_" + sample_id.split('_')[2]

            #print(block[-1])
            train_blocks = random.sample([0,1,2,3,4,5,6,7], config.set_in_train)

            if int(block[-1]) in train_blocks:

                train_X.append(features[:-1])
                if randomized is False:
                    block_label = 1 if "NR" in block else 0
                    train_y.append(block_label)
                else:
                    train_y.append(random.choice([0,1]))
            else:

                test_X.append(features[:-1])
                if randomized is False:
                    block_label = 1 if "NR" in block else 0
                    test_y.append(block_label)
                else:
                    test_y.append(random.choice([0,1]))


    else:
        sys.exit("Classification task {0} not defined!".format(config.class_task))

    #print(len(train_X), len(test_X))
    #print(len(test_X), len(test_y))

    np.random.seed(seed_value)
    train_X, train_y = shuffle(train_X, train_y)

    # scale feature values
    scaling = MinMaxScaler(feature_range=(-1, 1)).fit(train_X)
    train_X = scaling.transform(train_X)
    test_X = scaling.transform(test_X)

    # train SVM classifier
    clf = SVC(random_state=seed_value, kernel=config.kernel, gamma='scale', cache_size=1000)
    clf.fit(train_X, train_y)
    predictions = clf.predict(test_X)
    accuracy = len([i for i, j in zip(predictions, test_y) if i == j]) / len(test_y)

    # get coefficients
    if config.kernel is 'linear':
        coefficients = clf.coef_
    else:
        coefficients = [[]]

    return predictions, test_y, accuracy, coefficients

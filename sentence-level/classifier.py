from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import random
import config
import sys


def svm(samples, seed_value, randomized=False):
    X = []
    y = []

    if config.class_task == "tasks":
        for sample_id, features in samples.items():
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
                if features[-1] == "Sess1" and len(y) < 150:
                    X.append(features[:-1])
                    y.append(1)
                elif features[-1] == "Sess2" and len(y) < 150:
                    X.append(features[:-1])
                    y.append(0)

            else:
                y.append(random.choice([0, 1]))

        print("Samples:")
        print(y.count(0))
        print(y.count(1))
        max_samples_current_subject = min(y.count(0), y.count(1))

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

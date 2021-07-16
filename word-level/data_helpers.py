from datetime import date
import numpy as np
import config
import h5py


def load_matlab_files(task, subject):
    """loads Matlab files depending on which files are required for the chosen classification task"""

    if task.startswith("sentiment"):
        filename_sr = config.rootdir_zuco1 + "results" + subject + "_SR.mat"
        f_sr = h5py.File(filename_sr, 'r')
        sentence_data_sr = f_sr['sentenceData']

        return [(f_sr, sentence_data_sr)]

    elif task.startswith('reldetect'):
        loaded_matlab_data = []
        if subject.startswith('Z'):  # subjects from ZuCo 1
            # todo: add Sentiment data?
            filename_nr = config.rootdir_zuco1 + "results" + subject + "_NR.mat"
            f_nr = h5py.File(filename_nr, 'r')
            sentence_data_nr = f_nr['sentenceData']
            loaded_matlab_data.append((f_nr, sentence_data_nr))
        elif subject.startswith('Y'):  # subjects from ZuCo 2
            filename_nr = config.rootdir_zuco2 + "results" + subject + "_NR.mat"
            f_nr = h5py.File(filename_nr, 'r')
            sentence_data_nr = f_nr['sentenceData']
            loaded_matlab_data.append((f_nr, sentence_data_nr))
        else:
            print("UNDEFINED SUBJECT NAME")

        return [(f_nr, sentence_data_nr)]

    elif task.startswith('ner'):
        loaded_matlab_data = []
        if subject.startswith('Z'):  # subjects from ZuCo 1
            # load NR + sentiment task from ZuCo 1
            filename_nr = config.rootdir_zuco1 + "results" + subject + "_NR.mat"
            f_nr = h5py.File(filename_nr, 'r')
            sentence_data_nr = f_nr['sentenceData']
            loaded_matlab_data.append((f_nr, sentence_data_nr))
            filename_sr = config.rootdir_zuco1 + "results" + subject + "_SR.mat"
            f_sr = h5py.File(filename_sr, 'r')
            sentence_data_sr = f_sr['sentenceData']
            loaded_matlab_data.append((f_sr, sentence_data_sr))
        elif subject.startswith('Y'):  # subjects from ZuCo 1
            # load NR task from ZuCo 2
            filename_nr = config.rootdir_zuco2 + "results" + subject + "_NR.mat"
            f_nr = h5py.File(filename_nr, 'r')
            sentence_data_nr = f_nr['sentenceData']
            loaded_matlab_data.append((f_nr, sentence_data_nr))
        else:
            print("UNDEFINED SUBJECT NAME")

        return loaded_matlab_data

    elif task.startswith('read-task') or task.startswith('tasks-cross-subj'):
        loaded_matlab_data = []
        if subject.startswith('Z'):  # subjects from ZuCo 1
            # load NR + sentiment task from ZuCo 1
            label = "NR"
            filename_nr = config.rootdir_zuco1 + "results" + subject + "_NR.mat"
            f_nr = h5py.File(filename_nr, 'r')
            sentence_data_nr = f_nr['sentenceData']
            loaded_matlab_data.append((f_nr, sentence_data_nr, label))
            label = "TSR"
            filename_sr = config.rootdir_zuco1 + "results" + subject + "_TSR.mat"
            f_sr = h5py.File(filename_sr, 'r')
            sentence_data_sr = f_sr['sentenceData']
            loaded_matlab_data.append((f_sr, sentence_data_sr, label))
        elif subject.startswith('Y'):  # subjects from ZuCo 2
            # load NR task from ZuCo 2
            filename_nr = config.rootdir_zuco2 + "results" + subject + "_NR.mat"
            label = "NR"
            f_nr = h5py.File(filename_nr, 'r')
            sentence_data_nr = f_nr['sentenceData']
            loaded_matlab_data.append((f_nr, sentence_data_nr, label))
            filename_sr = config.rootdir_zuco2 + "results" + subject + "_TSR.mat"
            label = "TSR"
            f_sr = h5py.File(filename_sr, 'r')
            sentence_data_sr = f_sr['sentenceData']
            loaded_matlab_data.append((f_sr, sentence_data_sr, label))
        else:
            print("UNDEFINED SUBJECT NAME")

        return loaded_matlab_data


def save_results(fold_results_dict, task, subj="all"):
    """aggegates the results in fold_results_dict over all folds and
    saves hyper-parameters and results to a result file"""

    result_file = open('results/'+str(date.today()) + "_results_" + task + "_" + "-".join(config.feature_set) + subj + "_sacc" + str(config.saccades) + ".txt", 'a')

    # print header
    print("lstm_dim", "lstm_layers", "dense_dim", "dropout", "batch_size", "epochs", "lr", "embedding_type",
          "random_seed", "train_acc", "val_acc", "test_acc", "test_std", "avg_precision", "std_precision",
          "avg_recall", "std_recall", "avg_fscore", "std_fscore", "threshold", "folds", "training_time", 'best_ep', 'patience', 'min_delta', "model", file=result_file)

    # training scores
    train_acc = np.mean([ep[-1] for ep in fold_results_dict['train-accuracy']])

    # validation scores
    val_acc = np.mean([ep[-1] for ep in fold_results_dict['val-accuracy']])

    # test scores
    avg_accuracy = np.mean(fold_results_dict['test-accuracy'])
    avg_precision = np.mean(fold_results_dict['precision'])
    avg_recall = np.mean(fold_results_dict['recall'])
    avg_fscore = np.mean(fold_results_dict['fscore'])
    std_accuracy = np.std(fold_results_dict['test-accuracy'])
    std_precision = np.std(fold_results_dict['precision'])
    std_recall = np.std(fold_results_dict['recall'])
    std_fscore = np.std(fold_results_dict['fscore'])
    threshold = fold_results_dict['threshold'] if 'threshold' in fold_results_dict else "-"
    best_eps = ",".join(map(str, fold_results_dict['best-e']))
    folds = config.folds

    print(" ".join(map(str, fold_results_dict['params'])),train_acc, val_acc, avg_accuracy, std_accuracy, avg_precision,
          std_precision, avg_recall, std_recall, avg_fscore, std_fscore, threshold, folds, fold_results_dict['training_time'], best_eps, fold_results_dict['patience'], fold_results_dict['min_delta'], fold_results_dict['model'][-1], file=result_file)



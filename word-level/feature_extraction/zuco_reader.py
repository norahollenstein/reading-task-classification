import csv
import config
from . import gaze_extractor
from . import text_extractor
from . import eeg_extractor


# Read matlab files and extract gaze and/or EEG features


def extract_features(sent_data, feature_set, feature_dict, eeg_dict, gaze_dict, label_dict):
    """Extract features from ZuCo Matlab files"""

    # extract text for all models
    text_extractor.extract_sentences(sent_data, feature_dict, label_dict)

    #if 'sent_eeg_raw' in feature_set or 'combi_concat' in feature_set:
     #   eeg_extractor.extract_sent_raw_eeg(sent_data, eeg_dict)

    #if 'eeg_raw' in feature_set or 'combi_eeg_raw' in feature_set:
     #   eeg_extractor.extract_word_raw_eeg(sent_data, eeg_dict)

    if 'eeg_theta' in feature_set or 'eeg_alpha' in feature_set or 'eeg_beta' in feature_set or 'eeg_gamma' in feature_set:
        eeg_extractor.extract_word_band_eeg(sent_data, eeg_dict, label_dict)

    #if 'sent_eeg_theta' in feature_set or 'sent_eeg_alpha' in feature_set or 'sent_eeg_beta' in feature_set or 'sent_eeg_gamma' in feature_set:
     #   eeg_extractor.extract_sent_freq_eeg(sent_data, eeg_dict)

    #if "eye_tracking" in feature_set:# or 'combi_eye_tracking' in feature_set:
       # gaze_extractor.word_level_et_features(sent_data, gaze_dict, label_dict)

    #if 'fix_eeg_alpha' in feature_set or 'fix_eeg_theta' in feature_set or 'fix_eeg_gamma' in feature_set or 'fix_eeg_beta' in feature_set:
     #   eeg_extractor.extract_fix_band_eeg(sent_data, eeg_dict)


def extract_labels(feature_dict, label_dict, task, subject):
    """Get ground truth labels for all tasks"""

    if config.class_task == "read-task":

        count = 0
        #label_names = {'0': 2, '1': 1, '-1': 0}
        # (-1, negative) = 0,  (0, neutral) = 2, (1, positive) = 1
        i = 0

        if subject.startswith('Z'):  # subjects from ZuCo 1
            with open(config.base_dir+'eego/feature_extraction/labels/sentiment_sents_labels-corrected.txt', 'r') as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=';')
                for row in csv_reader:
                    sent = row[1]
                    label = row[-1]

                    if label not in label_names:
                        label_names[label] = i
                        i += 1

                    if sent in feature_dict:
                        label_dict[sent] = label_names[label]
                    else:
                        print("Sentence not found in feature dict!")
                        print(sent)
                        count += 1
                print('ZuCo 1 sentences not found:', count)

        else:
            print("Sentiment analysis only possible for ZuCo 1!!!")

    elif task == 'ner':

        count = 0
        label_names = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6}

        if subject.startswith('Z'):  # subjects from ZuCo 1
            # use NR + sentiment task from ZuCo 1
            ner_ground_truth = open(config.base_dir+'eego/feature_extraction/labels/zuco1_nr_ner.bio', 'r').readlines() + open(config.base_dir+'eego/feature_extraction/labels/zuco1_nr_sentiment_ner.bio', 'r').readlines()

            sent_tokens = []
            sent_labels = []
            for line in ner_ground_truth:

                # start of new sentence
                if line == '\n':
                    if sent_tokens in feature_dict.values():
                        sent_str = list(feature_dict.keys())[list(feature_dict.values()).index(sent_tokens)]
                        label_dict[sent_str] = [label_names[s] for s in sent_labels]
                    else:
                        print("Sentence not found in feature dict!")
                        print(sent_tokens)
                        count += 1

                    sent_tokens = []
                    sent_labels = []
                else:
                    line = line.split('\t')
                    sent_tokens.append(line[0])
                    sent_labels.append(line[1].strip())

            print('ZuCo 1 sentences not found:', count)

        if subject.startswith('Y'):  # subjects from ZuCo 2
            # use NR task from ZuCo 2
            ner_ground_truth = open(config.base_dir+'eego/feature_extraction/labels/zuco2_nr_ner.bio', 'r').readlines()

            sent_tokens = []
            sent_labels = []
            for line in ner_ground_truth:

                # start of new sentence
                if line == '\n':
                    if sent_tokens in feature_dict.values():
                        sent_str = list(feature_dict.keys())[list(feature_dict.values()).index(sent_tokens)]
                        label_dict[sent_str] = [label_names[s] for s in sent_labels]
                    else:
                        print("Sentence not found in feature dict!")
                        print(sent_tokens)
                        count += 1

                    sent_tokens = []
                    sent_labels = []
                else:
                    line = line.split('\t')
                    sent_tokens.append(line[0])
                    sent_labels.append(line[1].strip())

            print('ZuCo 2 sentences not found:', count)

    elif task == 'reldetect':

        count = 0
        label_names = ["Visited", "Founder", "Nationality", "Wife", "PoliticalAffiliation", "JobTitle", "Education",
                          "Employer", "Awarded", "BirthPlace", "DeathPlace"]

        if subject.startswith('Z'):  # subjects from ZuCo 1
            # use NR + sentiment task from ZuCo 1
            ner_ground_truth = open(config.base_dir + 'eego/feature_extraction/labels/zuco1_nr_rel.bio',
                                    'r').readlines()

            for line in ner_ground_truth:

                line = line.split("\t")
                sent_str = line[0]
                labels = [int(l.strip()) for l in line[1:]]

                if sent_str in feature_dict:
                    label_dict[sent_str] = labels
                else:
                    print("Sentence not found in feature dict!")
                    print(sent_str)
                    count += 1

            print('ZuCo 1 sentences not found:', count)

        if subject.startswith('Y'):  # subjects from ZuCo 2
            # use NR task from ZuCo 2
            ner_ground_truth = open(config.base_dir + 'eego/feature_extraction/labels/zuco2_nr_rel.bio',
                                    'r').readlines()

            for line in ner_ground_truth:

                line = line.split("\t")
                sent_str = line[0]
                labels = [int(l.strip()) for l in line[1:]]

                if sent_str in feature_dict:
                    label_dict[sent_str] = labels
                else:
                    print("Sentence not found in feature dict!")
                    print(sent_str)
                    count += 1

            print('ZuCo 2 sentences not found:', count)






import matplotlib.pyplot as plt
import os
import numpy as np
import config
#from transformers import BertTokenizer
#from transformers import TFBertModel
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
from matplotlib import cm


def plot_prediction_distribution(true, pred):
    """Analyze label distribution of dataset"""

    plt.hist(true, bins=len(set(true)), color='green', alpha=0.5)
    plt.hist(pred, bins=len(set(pred)), color='blue', alpha=0.5)
    plt.savefig('pred-label-distribution-' + config.class_task + '.png')
    plt.tight_layout()
    plt.clf()


def plot_label_distribution(y):
    """Analyze label distribution of dataset"""

    if config.class_task == "reldetect":

        label_names = ["Visited", "Founder", "Nationality", "Wife", "Political Affiliation", "Job Title", "Education",
                       "Employer", "Awarded", "Birth Place", "Death Place"]

        all_relations = np.sum(y, 0)

        plt.clf()
        fig, ax = plt.subplots()
        all_relations, label_names = zip(*sorted(zip(all_relations, label_names)))

        cmap = cm.viridis(np.linspace(0, 1, len(all_relations)))
        ax.barh(range(len(all_relations)), all_relations, color=cmap)

        ax.set_xticks([0, 50, 100, 150, 200])
        ax.set_xticklabels([0, 50, 100, 150, 200], fontsize=12)
        ax.set_yticks(range(len(all_relations)))
        ax.set_yticklabels(labels=label_names, fontsize=12)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        fig.tight_layout()
        plt.savefig('label-distribution-' + config.class_task + '.png')
        fig.clf()

        # plot number of relation types per sentence
        rels_per_sentence = {}
        fig, ax = plt.subplots(figsize=(6,3))
        for s in y:
            if sum(s) not in rels_per_sentence:
                rels_per_sentence[sum(s)] = 1
            else:
                rels_per_sentence[sum(s)] += 1
        rels_sorted = sorted(rels_per_sentence.items(), reverse=True)
        rels = [i[0] for i in rels_sorted]
        sents = [i[1] for i in rels_sorted]
        ax.barh(rels, sents, color=cmap[:len(rels)])
        fig.gca().invert_yaxis()
        ax.set_ylabel('no. of relations', fontsize=12)
        ax.set_xlabel('no. of sentences', fontsize=12)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

        fig.tight_layout()
        fig.savefig('relation-distribution-' + config.class_task + '.png')
        fig.clf()

    else:
        fig, ax = plt.subplots(figsize=(6, 2))
        counts = []
        for cl in set(y):
            class_count = y.count(cl)
            counts.append(class_count)
        cmap = cm.viridis(np.linspace(0, 1, 11))
        ax.barh(list(set(y)), counts, color=cmap[:len(counts)])
        ax.set_yticks(range(len(counts)))
        ax.set_yticklabels(labels=["negative", "positive", "neutral"], fontsize=11)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.set_xlabel('no. of sentences')
        fig.tight_layout()
        plt.savefig('label-distribution-' + config.class_task + '.png')
        fig.clf()


def load_glove_embeddings(vocab_size, word_index, EMBEDDING_DIM):

    print('Indexing word vectors.')

    embeddings_index = {}
    with open(os.path.join(
            config.base_dir + 'eego/feature_extraction/embeddings/glove.6B.'+str(EMBEDDING_DIM)+'d.txt')) as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, 'f', sep=' ')
            embeddings_index[word] = coefs

    print('Found %s word vectors.' % len(embeddings_index))

    # prepare embedding matrix
    num_words = min(vocab_size, len(word_index) + 1)

    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i >= vocab_size:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    print("glove matrix:", embedding_matrix.shape)
    return embedding_matrix

"""
def get_bert_max_len(X):

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

    max_len = 0
    # For every sentence...
    for sent in X:
        # Tokenize the text and add [CLS] and [SEP] tokens.
        input_ids = tokenizer.encode(sent, add_special_tokens=True)
        # Update the maximum sentence length.
        max_len = max(max_len, len(input_ids))
    print('Max Bert sentence length: ', max_len)

    return max_len


def prepare_sequences_for_bert_with_mask(X, max_length):
    
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    # Tokenize all of the sentences and map the tokens to their word IDs.
    input_ids = []
    attention_masks = []

    # For every sentence...
    for sent in X:
        # `encode_plus` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        #   (5) Pad or truncate the sentence to `max_length`
        #   (6) Create attention masks for [PAD] tokens.
        encoded_dict = tokenizer.encode_plus(
                            sent,                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            max_length = max_length,           # Pad & truncate all sentences.
                            pad_to_max_length = True,
                            return_attention_mask = True,   # Construct attn. masks.
                       )
        
        # Add the encoded sentence to the list.    
        input_ids.append(encoded_dict['input_ids'])
        
        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])

    # Convert the lists into tensors.
    input_ids = np.vstack(input_ids)
    attention_masks = np.vstack(attention_masks)

    return input_ids, attention_masks


def create_new_bert_layer():
    bert = TFBertModel.from_pretrained("bert-base-uncased")
    return bert
"""

def scale_feature_values(X):
    """Scale eye-tracking and EEG feature values"""

    scaler = MinMaxScaler(feature_range=(0, 1))
    feat_values = []

    for feat in range(len(X[0][0])):
        for sentence in X:
            for token in sentence:
                feat_values.append(token[feat])

    # train the normalization
    feat_values = np.array(feat_values).reshape(-1, 1)
    scaler = scaler.fit(feat_values)
    print('Min: %f, Max: %f' % (scaler.data_min_, scaler.data_max_))
    # normalize the dataset and print
    normalized = scaler.transform(feat_values)

    # add normalized values back to feature list
    i = 0
    for sentence in X:
        for token in sentence:
            token[feat] = normalized[i]
            i += 1

    return X


def plot_confusion_matrix(cm):
    import matplotlib.pyplot as plt
    from mlxtend.plotting import plot_confusion_matrix

    fig, ax = plot_confusion_matrix(conf_mat=cm, colorbar=True,
                                    show_absolute=True,
                                    show_normed=True)
    plt.title("Confusion matrix: " + config.class_task + ", " + config.feature_set[0])
    plt.savefig("CM_" + config.class_task + "_" + config.feature_set[0] + ".pdf")
    plt.clf()
    # plt.show()


def prepare_text(X_text, embedding_type, random_seed):

    np.random.seed(random_seed)

    vocab_size = 100000

    # prepare text samples
    print('Processing text dataset...')

    print('Found %s sentences.' % len(X_text))

    tokenizer = Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(X_text)
    sequences = tokenizer.texts_to_sequences(X_text)
    max_length_text = max([len(s) for s in sequences])
    print("Maximum sentence length: ", max_length_text)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    num_words = min(vocab_size, len(word_index) + 1)

    if embedding_type is 'none':
        X_data_text = pad_sequences(sequences, maxlen=max_length_text, padding='post', truncating='post')
        print('Shape of data tensor:', X_data_text.shape)

        return X_data_text, num_words, ""

    if embedding_type is 'glove':
        X_data_text = pad_sequences(sequences, maxlen=max_length_text, padding='post', truncating='post')
        print('Shape of data tensor:', X_data_text.shape)

        print("Loading Glove embeddings...")
        embedding_dim = 300
        embedding_matrix = load_glove_embeddings(vocab_size, word_index, embedding_dim)

        return X_data_text, num_words, embedding_matrix

    """
    if embedding_type is 'bert':
        print("Prepare sequences for Bert ...")
        max_length = get_bert_max_len(X_text)
        X_data_text, X_data_masks = prepare_sequences_for_bert_with_mask(X_text, max_length)
        print('Shape of data tensor:', X_data_text.shape)
        print('Shape of data (masks) tensor:', X_data_masks.shape)

        return X_data_text, num_words, X_data_masks
    """


def prepare_cogni_seqs(cogni_dict):
    print('Processing cognitive data...')
    # prepare cognitive data
    cogni_X = []
    max_length_cogni = 0
    # average cognitive features over all subjects
    for s in cogni_dict.values():
        sent_feats = []
        max_length_cogni = max(len(s), max_length_cogni)
        for w, fts in s.items():
            subj_mean_word_feats = np.nanmean(fts, axis=0)
            subj_mean_word_feats[np.isnan(subj_mean_word_feats)] = 0.0
            sent_feats.append(subj_mean_word_feats)
        cogni_X.append(sent_feats)

    return cogni_X, max_length_cogni


def pad_cognitive_feature_seqs(eeg_X, max_length_cogni, modality):
    for s in eeg_X:
        while len(s) < max_length_cogni:
            if modality == "eeg":
                # 105 = number of EEG electrodes
                s.append(np.zeros(105))
            elif modality == "eye_tracking":
                # 5 = number of gaze features
                if config.saccades is True:
                    s.append(np.zeros(9))
                else:
                    s.append(np.zeros(5))
            else:
                print("No EEG or eye-tracking features specified in config file!")

    X_data = np.array(eeg_X)
    return X_data


def callbacks(fold, random_seed_value):
    """Define Keras callbacks for early stopping and saving model at best epoch"""

    es = EarlyStopping(monitor='val_accuracy', mode='max', min_delta=config.min_delta, patience=config.patience)

    d = datetime.datetime.now()
    model_name = 'model-checkpoints/' + str(random_seed_value) + '_fold' + str(fold) + '_' + config.class_task + '_' + config.feature_set[0] + '_' + config.embeddings + '_' + d.strftime("%d%m%Y-%H:%M:%S") + '.h5'
    mc = ModelCheckpoint(model_name, monitor='val_accuracy', mode='max', save_weights_only=True, save_best_only=True, verbose=1)

    return es, mc, model_name


def drop_train_sents(sample_list):

    to_delete = round(len(sample_list[0]) * config.data_percentage)
    print("Deleting first " + str(to_delete) + " training samples")

    for idx, li in enumerate(sample_list):
        li = li[to_delete:]
        sample_list[idx] = li

    return sample_list


def drop_classes(y):

    # tested with dropping the 4, 6 or 8 least frequent relations
    print("Deleting least frequent " + str(len(config.drop_classes)) + " classes")

    new_y = []
    for idx, sample in enumerate(y):
        sample = [i for j, i in enumerate(sample) if j not in config.drop_classes]
        new_y.append(sample)

    return new_y


def drop_samples(y, X, X_eeg=None):

    # tested with dropping the 4, 6 or 8 least frequent relations
    label_names = ["Visited", "Founder", "Nationality", "Wife", "Political Affiliation", "Job Title", "Education",
                   "Employer", "Awarded", "Birth Place", "Death Place"]
    rel_index = 7
    print(label_names[rel_index], " vs. no relation - binary classification")

    # sample[5] = job_title
    # sample[0] = visited
    # 7 = employer

    print("taking only ", label_names[rel_index], " samples.")


    new_y = []
    new_X = []
    new_X_eeg = []
    for idx, sample in enumerate(y):

        job_title = True if int(sample[rel_index]) == 1 else False
        no_rel = True if all(int(n) == 0 for n in sample) else False
        if job_title:
            new_y.append(np.array([1.,0.]))
            new_X.append(X[idx])
            if X_eeg is not None:
                new_X_eeg.append(X_eeg[idx])
        if no_rel:
            new_y.append(np.array([0., 1.]))
            new_X.append(X[idx])
            if X_eeg is not None:
                new_X_eeg.append(X_eeg[idx])

    if not new_X_eeg:
        return new_y, new_X
    else:
        return new_y, new_X, new_X_eeg
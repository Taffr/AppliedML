import os
import pickle
import embeddings_parser
import numpy as np
from CoNLLDictorizer import CoNLLDictorizer
from keras.preprocessing.sequence import  pad_sequences
from keras.utils import to_categorical

EMBEDDING_DIM = 100

def parse_corpus(file_path):
    save_file_path = 'pickled/train_vector.p'
    if os.path.isfile(save_file_path):
        train = pickle.load(open(save_file_path, 'rb'))
    else:
        corpus = open(file_path).read().strip()
        column_names = ['form', 'pos', 'syntatic_chunk', 'named_entity']
        dictorizer = CoNLLDictorizer(column_names, '\n\n', ' ')

        train = dictorizer.transform(corpus)
        pickle.dump(train, open(save_file_path, 'wb'))

    X = []
    Y = []
    for sentence in train:
        x = []
        y = []
        for word in sentence:
            x.append(word['form'].lower())
            y.append(word['named_entity'])
        X.append(x)
        Y.append(y)


    return X, Y

def extract_unique_words_pos(X_train, Y_train):
    unique_words = set()
    for sentence in X_train:
        for word in sentence:
            unique_words.add(word)
    unique_words = sorted(list(unique_words))

    unique_pos = set()
    for sentence in Y_train:
        for pos in sentence:
            unique_pos.add(pos)
    unique_pos = sorted(list(unique_pos))
    return unique_words, unique_pos

def combine_train_and_embeddings_words(train, embeddings):
    words = sorted(list(set(train + list(embeddings))))

    return words


def create_index_maps(words, pos):
    word_indices = dict(enumerate(words, start=2))
    pos_indices = dict(enumerate(pos, start=2))

    word_to_index = {v: k for k, v in word_indices.items()}
    pos_to_index = {v: k for k, v in pos_indices.items()}

    return word_to_index, pos_to_index

def build_embedding_matrix(words, embeddings, word_to_index):
    embedding_matrix = np.random.uniform(-0.05, 0.05, (len(words) + 2, EMBEDDING_DIM))

    for word in words:
        if word in embeddings:
            embedding_matrix[word_to_index[word]] = embeddings[word]

    return embedding_matrix

def convert_to_sequences(X_train, Y_train, word_to_index, pos_to_index):
    X_sequence = []
    Y_sequence = []

    for sentence in X_train:
        x = [word_to_index[word] for word in sentence]
        X_sequence.append(x)

    for sentence in Y_train:
        y = [pos_to_index[pos] for pos in sentence]
        Y_sequence.append(y)

    X_sequence = pad_sequences(X_sequence, maxlen=150)

    # print(X_sequence[1])
    # print(pad_sequences(Y_sequence, maxlen=150)[1])

    Y_sequence = to_categorical(pad_sequences(Y_sequence, maxlen=150))

    return X_sequence, Y_sequence


def to_index(X, to_index_dict):
    indicies = []
    for x in X:
        x_idx = list(map(lambda x: to_index_dict.get(x, 1), x))
        indicies += [x_idx]

    return indicies


def preprocess():
    X_train, Y_train = parse_corpus('./data/NER-data/eng.train')
    unique_train, unique_pos = extract_unique_words_pos(X_train, Y_train)

    embeddings = embeddings_parser.read_glove_embeddings('./data/glove.6B.100d.txt')
    words = combine_train_and_embeddings_words(unique_train, embeddings)

    word_to_index, pos_to_index = create_index_maps(words, unique_pos)

    embedding_matrix = build_embedding_matrix(words, embeddings, word_to_index)

    X_sequence, Y_sequence = convert_to_sequences(X_train, Y_train, word_to_index, pos_to_index)

    return X_sequence, Y_sequence, words, unique_pos, embedding_matrix, word_to_index, pos_to_index



if __name__ == '__main__':
    preprocess()
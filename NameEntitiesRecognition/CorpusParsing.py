import pickle
import sys
import os
from sklearn.feature_extraction import DictVectorizer
import time
import regex as re
from keras import models, layers
import sys
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from keras.models import load_model
import math
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import LSTM, Bidirectional, SimpleRNN, Dense


def parse_corpus(file_path):
    corpus = open(file_path).read()
    sentences = re.split('\n\n', corpus)

    X = []
    Y = []
    for sentence in sentences:
        x = []
        y = []
        words = sentence.split('\n')
        for word in words:
            word_parts = word.split()
            if len(word_parts) == 0:
                continue

            x.append(word_parts[0].lower())
            y.append(word_parts[len(word_parts) - 1])
        X.append(x)
        Y.append(y)

    return X, Y


def vocabulary(pickle_path, sentences):
    glove = pickle.load(open(pickle_path, 'rb'))
    words = {}

    for word in glove:
        words[word] = 0

    for sentence in sentences:
        for word in sentence:
            words[word] = 0

    return list(words.keys())


def index_sequences(vocabulary, tags):
    # print(vocabulary)
    idx_word = dict(enumerate(vocabulary, start=2))
    idx_pos = dict(enumerate(tags, start=2))

    # word_idx = {v: k for k in idx_word.keys() for v in idx_word[k]}

    word_idx = {v: k for k, v in idx_word.items()}
    pos_idx = {v: k for k, v in idx_pos.items()}
    # print(tags)
    # print(idx_pos)

    # print(word_idx)


def to_index(X, idx):
    """
    Convert the word lists (or POS lists) to indexes
    :param X: List of word (or POS) lists
    :param idx: word to number dictionary
    :return:
    """
    X_idx = []
    for x in X:
        # We map the unknown words to one
        x_idx = list(map(lambda x: idx.get(x, 1), x))
        X_idx += [x_idx]
    return X_idx


if __name__ == '__main__':
    X, Y = parse_corpus('./data/NER-data/eng.train')
    words = vocabulary('embeddings.p', X)
    # index_sequences(words, Y)

    vocabulary_words = sorted(list(
        set([word for sentence
             in X for word in sentence])))
    pos = sorted(list(set([pos for sentence
                           in Y for pos in sentence])))

    embeddings_dict = pickle.load(open('embeddings.p', 'rb'))
    embeddings_words = embeddings_dict.keys()
    print('Words in GloVe:', len(embeddings_dict.keys()))
    vocabulary_words = sorted(list(set(vocabulary_words +
                                       list(embeddings_words))))
    cnt_uniq = len(vocabulary_words) + 2
    print('# unique words in the vocabulary: embeddings and corpus:',
          cnt_uniq)

    # We start at one to make provision for the padding symbol 0
    # in RNN and LSTMs and 1 for the unknown words
    idx_word = dict(enumerate(vocabulary_words, start=2))
    idx_pos = dict(enumerate(pos, start=2))
    word_idx = {v: k for k, v in idx_word.items()}
    pos_idx = {v: k for k, v in idx_pos.items()}
    print('word index:', list(word_idx.items())[:10])
    print('POS index:', list(pos_idx.items())[:10])

    # We create the parallel sequences of indexes
    X_idx = to_index(X, word_idx)
    Y_idx = to_index(Y, pos_idx)
    print('First sentences, word indices', X_idx[:3])
    print('First sentences, POS indices', Y_idx[:3])

    embedding_matrix = np.random.rand(cnt_uniq, 100)

    for word in vocabulary_words:
        if word in embeddings_dict:
            # If the words are in the embeddings, we fill them with a value
            embedding_matrix[word_idx[word]] = embeddings_dict[word]

    print('Shape of embedding matrix:', embedding_matrix.shape)
    print('Embedding of table', embedding_matrix[word_idx['table']])
    print('Embedding of the padding symbol, idx 0, random numbers',
          embedding_matrix[0])

    X = pad_sequences(X_idx)
    Y = pad_sequences(Y_idx)

    print(X[0])
    print(Y[0])

    # The number of POS classes and 0 (padding symbol)
    Y_train = to_categorical(Y, num_classes=len(pos) + 2)
    Y_train[0]

    model = models.Sequential()
    model.add(layers.Embedding(len(vocabulary_words) + 2,
                               100,
                               mask_zero=True,
                               input_length=None))
    model.layers[0].set_weights([embedding_matrix])
    # The default is True
    model.layers[0].trainable = True
    # model.add(SimpleRNN(100, return_sequences=True))
    # model.add(Bidirectional(SimpleRNN(100, return_sequences=True)))
    model.add(Bidirectional(LSTM(100, return_sequences=True)))
    model.add(Dense(len(pos) + 2, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])
    model.summary()
    model.fit(X, Y_train, epochs=2, batch_size=128)

    model.save('model_1.h5')

    #
    # # In X_dict, we replace the words with their index
    # X_test_cat, Y_test_cat = build_sequences(test_dict)
    # # We create the parallel sequences of indexes
    # X_test_idx = to_index(X_test_cat, word_idx)
    # Y_test_idx = to_index(Y_test_cat, pos_idx)
    #
    # print('X[0] test idx', X_test_idx[0])
    # print('Y[0] test idx', Y_test_idx[0])
    #
    # X_test_padded = pad_sequences(X_test_idx)
    # Y_test_padded = pad_sequences(Y_test_idx)
    # print('X[0] test idx passed', X_test_padded[0])
    # print('Y[0] test idx padded', Y_test_padded[0])
    # # One extra symbol for 0 (padding)
    # Y_test_padded_vectorized = to_categorical(Y_test_padded,
    #                                           num_classes=len(pos) + 2)
    # print('Y[0] test idx padded vectorized', Y_test_padded_vectorized[0])
    # print(X_test_padded.shape)
    # print(Y_test_padded_vectorized.shape)




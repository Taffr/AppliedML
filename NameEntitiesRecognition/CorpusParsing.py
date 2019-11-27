import os
import pickle
import regex as re
import CoNLLDictorizer


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


if __name__ == '__main__':
    X, Y = parse_corpus('./data/NER-data/eng.train')
    words = vocabulary('embeddings.p', X)
    # index_sequences(words, Y)

    vocabulary_words = sorted(list(
        set([word for sentence
             in X for word in sentence])))
    pos = sorted(list(set([pos for sentence
                           in Y for pos in sentence])))

    print(pos)

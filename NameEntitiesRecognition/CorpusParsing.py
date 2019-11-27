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


if __name__ == '__main__':
    X, Y = parse_corpus('./data/NER-data/eng.train')
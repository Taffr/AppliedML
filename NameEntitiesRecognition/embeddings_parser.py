import os
import pickle
import numpy as np
from numpy.linalg import norm

def read_glove_embeddings(file_path):
    save_file_path = 'pickled/embeddings.p'
    if (os.path.isfile(save_file_path)):
        print('Loads embeddings from file...')
        embeddings = pickle.load(open(save_file_path, 'rb'))
    else:
        embeddings = {}
        print('Creates a new embeddings dictionary...')
        glove = open(file_path)
        for line in glove:
            values = line.strip().split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings[word] = coefs

        glove.close()
        pickle.dump(embeddings, open(save_file_path, 'wb'))

    return embeddings


def five_closest_words(embeddings):
    embedding_table = embeddings['table']
    embedding_france = embeddings['france']
    embedding_sweden = embeddings['sweden']

    norm_table = norm(embedding_table)
    norm_france = norm(embedding_france)
    norm_sweden = norm(embedding_sweden)

    similarities_table = {}
    similarities_france = {}
    similarities_sweden = {}
    for w in embeddings:
        norm_w = np.linalg.norm(embeddings[w])

        similarities_table[w] = embedding_table.dot(embeddings[w]) / (norm_table * norm_w)
        similarities_france[w] = embedding_france.dot(embeddings[w]) / (norm_france * norm_w)
        similarities_sweden[w] = embedding_sweden.dot(embeddings[w]) / (norm_sweden * norm_w)


    # Sort the dictionaries and get top 5
    # Excludes the first element, since this will be the same as the word we compared with,
    # i.e. 'table', 'france' and 'sweden'.
    top_5_table = sorted(similarities_table, key=similarities_table.get, reverse=True)[1:6]
    top_5_france = sorted(similarities_france, key=similarities_france.get, reverse=True)[1:6]
    top_5_sweden = sorted(similarities_sweden, key=similarities_sweden.get, reverse=True)[1:6]

    print('table: ', top_5_table)
    print('france: ', top_5_france)
    print('sweden: ', top_5_sweden)

if __name__ == '__main__':
    embeddings = read_glove_embeddings('./data/glove.6B.100d.txt')
    five_closest_words(embeddings)

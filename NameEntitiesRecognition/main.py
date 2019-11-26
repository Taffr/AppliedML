import os
import pickle
import numpy as np

def read_glove_embeddings(file_path):
    save_file_path = 'embeddings.p'
    embeddings = {}

    if (os.path.isfile(save_file_path)):
        print('Loads embeddings from file...')
        embeddings = pickle.load(open(save_file_path, 'rb'))
    else:
        print('Creates a new embeddings dictionary...')
        glove = open(file_path)
        for line in glove:
            values = line.strip().split()
            word = values[0]
            embeddings[word] = np.array(values[1:])

        glove.close()
        pickle.dump(embeddings, open(save_file_path, 'wb'))

    return embeddings


def five_closest_words(embeddings):
    print('table', embeddings['table'])
    print('france', embeddings['france'])
    print('sweden', embeddings['sweden'])

if __name__ == '__main__':
    embeddings = read_glove_embeddings('./data/glove.6B.100d.txt')
    five_closest_words(embeddings)
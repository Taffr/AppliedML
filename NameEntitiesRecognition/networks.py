import os
import preprocessing
import pickle
import numpy as np
from keras import models, layers
from keras.layers import LSTM, Bidirectional, SimpleRNN, Dense, Dropout
import matplotlib.pyplot as plt
from keras.models import load_model

EMBEDDING_DIM = 100

def build_network(word_count, pos_count, embedding_matrix, network_type, use_bidirectional):
    model = models.Sequential()

    model.add(layers.Embedding(word_count + 2,
                               EMBEDDING_DIM,
                               mask_zero=True,  # input value 0 is a special padding value
                               input_length=None))
    model.layers[0].set_weights([embedding_matrix])
    model.layers[0].trainable = True

    if network_type.lower() == 'rnn':
        network = SimpleRNN(100, return_sequences=True)
    elif network_type.lower() == 'lstm':
        network = LSTM(100, return_sequences=True)

    if use_bidirectional:
        model.add(Bidirectional(network))
    else:
        model.add(network)

    model.add(Dropout(0.2))
    model.add(Dense(pos_count + 2, activation='softmax'))

    return model

def fit_model(model, X_train, Y_train, epochs = 2, batch_size = 128):
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])

    model.summary()
    history = model.fit(X_train, Y_train, epochs = epochs, batch_size = batch_size)
    return history

def save_history_metrics(history, save_file_base_path):
    acc = history.history['acc']
    loss = history.history['loss']
    # val_acc = history.history['val_acc']
    # val_loss = history.history['val_loss']

    f_out = open(save_file_base_path + '_metrics.txt', 'w')
    f_out.write('** History metrics **\n')
    f_out.write('Training accuracy: ' + str(acc[len(acc) - 1]) + '\n')
    f_out.write('Training loss: ' + str(loss[len(loss) - 1]) + '\n')
    # f_out.write('Validation accuracy: ' + str(val_acc[len(val_acc) - 1]) + '\n')
    # f_out.write('Validation loss: ' + str(val_loss[len(val_loss) - 1]) + '\n')

    f_out.close()

    # epochs = range(len(acc))
    #
    # plt.plot(epochs, acc, 'bo', label='Training acc')
    # plt.plot(epochs, val_acc, 'b', label='Validation acc')
    # plt.title('Training and validation accuracy')
    # plt.legend()
    # plt.savefig(save_file_base_path + '_accuracy.png')
    #
    # plt.figure()
    #
    # plt.plot(epochs, loss, 'bo', label='Training loss')
    # plt.plot(epochs, val_loss, 'b', label='Validation loss')
    # plt.title('Training and validation loss')
    # plt.legend()
    # plt.savefig(save_file_base_path + '_loss.png')

    # plt.show()


if __name__ == '__main__':
    model_base_path = 'recurrent_lstm_with-bidirectional_5_epochs'
    X_train, Y_train, words, pos, embedding_matrix, word_to_index, pos_to_index, word_indicies, pos_indicies = preprocessing.preprocess()

    if os.path.isfile(model_base_path + '.h5') and os.path.isfile('pickled/' + model_base_path + '_history.p'):
        model = load_model(model_base_path + '.h5')
        history = pickle.load(open('pickled/' + model_base_path + '_history.p', 'rb'))
    else:

        # model = load_model('flower_classification_model_overfit.h5')

        model = build_network(len(words), len(pos), embedding_matrix, 'lstm', True)

        history = fit_model(model, X_train, Y_train, 5)

        pickle.dump(history, open('pickled/' + model_base_path + '_history.p', 'wb'))
        model.save(model_base_path + '.h5')

    # save_history_metrics(history, model_base_path)

    X_test, Y_test = preprocessing.parse_corpus('./data/NER-data/eng.test')
    # X_test_indices = preprocessing.to_index(X_test, word_to_index)
    # Y_test_indices = preprocessing.to_index(Y_test, pos_to_index)
    X_test_seq, Y_test_seq = preprocessing.convert_to_sequences(X_test, Y_test, word_to_index, pos_to_index)
    print(X_test_seq[1])
    print(Y_test_seq[1])

    test_loss, test_acc = model.evaluate(X_test_seq, Y_test_seq)

    print('test_loss: ', test_loss)
    print('test_acc: ', test_acc)

    pos_predictions = model.predict(X_test_seq)
    pos_pred_num = []
    for sent_nbr, sent_pos_predictions in enumerate(pos_predictions):
        pos_pred_num += [sent_pos_predictions[-len(X_test[sent_nbr]):]]

    pos_pred = []
    for sentence in pos_pred_num:
        pos_pred_idx = list(map(np.argmax, sentence))
        pos_pred_cat = list(map(lambda x: pos_indicies.get(x, 1), pos_pred_idx))
        # pos_pred_cat = list(map(pos_indicies.get(), pos_pred_idx))
        pos_pred += [pos_pred_cat]

    f_out = open('prediction.txt', 'w')

    for sentence in zip(X_test, Y_test, pos_pred):
        # print(sentence)
        for i in range(len(sentence[0])):
            f_out.write("\n" + str(sentence[0][i]) + " " + str(sentence[1][i]) + ' ' + str(sentence[2][i]))

    f_out.close()




import os
import pickle
import numpy as np
from keras.applications import InceptionV3
from keras.preprocessing.image import ImageDataGenerator
from keras import models, layers
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn import metrics

# Instantiate the InceptionV3 model
conv_base = InceptionV3(weights='imagenet', # specifies which weight checkpoint to initialize the model from
                        include_top=False, # should not include the densely-connected classifier on top of the network.
                        input_shape=(150, 150, 3))

conv_base.summary()

base_dir = "./datasets/flowers_split/"

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')
n_train = 2592
n_validation = 865
n_test = 866

datagen = ImageDataGenerator(rescale=1./255)
batch_size = 20
epochs = 30

def extract_features(directory, sample_count):
    features = np.zeros(shape=(sample_count, 3, 3, 2048))
    labels = np.zeros(shape=(sample_count, 5))
    generator = datagen.flow_from_directory(
        directory,
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='categorical'
    )
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size : (i + 1) * batch_size] = features_batch
        labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        i += 1
        # if (i % 25 == 0):
        #     print(i)
        if i * batch_size >= sample_count:
            break
    return features, labels

# train_features, train_labels = extract_features(train_dir, n_train)
# pickle.dump(train_features, open('train_features.p', 'wb'))
# pickle.dump(train_labels, open('train_labels.p', 'wb'))
#
# validation_features, validation_labels = extract_features(validation_dir, n_validation)
# pickle.dump(validation_features, open('validation_features.p', 'wb'))
# pickle.dump(validation_labels, open('validation_labels.p', 'wb'))
#
# test_features, test_labels = extract_features(test_dir, n_test)
# pickle.dump(test_features, open('test_features.p', 'wb'))
# pickle.dump(test_labels, open('test_labels.p', 'wb'))

train_features = pickle.load(open('train_features.p', 'rb'))
train_labels = pickle.load(open('train_labels.p', 'rb'))
validation_features = pickle.load(open('validation_features.p', 'rb'))
validation_labels = pickle.load(open('validation_labels.p', 'rb'))

train_features = np.reshape(train_features, (n_train, 3 * 3 * 2048))
validation_features = np.reshape(validation_features, (n_validation, 3 * 3 * 2048))

print(train_features.shape)
print(train_labels.shape)
print(validation_features.shape)
print(validation_labels.shape)


model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_dim=3 * 3 * 2048))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(5, activation='softmax'))

model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["acc"])

# train_labels = to_categorical(train_labels)
# validation_labels = to_categorical(validation_labels)

# history = model.fit(
#     train_features,
#     train_labels,
#     epochs=epochs,
#     batch_size=batch_size,
#     validation_data=(validation_features, validation_labels)
# )

# pickle.dump(history, open('history_pretrained_1.p', 'wb'))
# model.save('flower_classification_model_pretrained_1.h5')

history = pickle.load(open('history_pretrained_1.p', 'rb'))
model = models.load_model('flower_classification_model_pretrained_1.h5')

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


test_features = pickle.load(open('test_features.p', 'rb'))
test_labels = pickle.load(open('test_labels.p', 'rb'))
# test_features = to_categorical(test_features)
test_features = np.reshape(test_features, (n_test, 3 * 3 * 2048))
print(test_features.shape)
print(train_features.shape)


# model.predict(test_features, batch_size=1, verbose=0)
loss, acc = model.evaluate(test_features, test_labels, batch_size=1, verbose=0)

print('loss: ', loss)
print('acc: ', acc)

y_pred = model.predict(test_features, batch_size=1, verbose=0);

# print(y_pred)

y = [np.argmax(n) for n in y_pred]
labels = [np.argmax(n) for n in test_labels]
# print(y)
# print(test_labels)

# print(y_pred)
# print(y)
# print(labels)
print('\n*** Confusion matrix ***')
print(metrics.confusion_matrix(y, labels))
print('\n*** Classification report ***')
print(metrics.classification_report(y, labels))
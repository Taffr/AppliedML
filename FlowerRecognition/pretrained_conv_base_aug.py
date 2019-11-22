import os, pickle
from keras import models
from keras import layers
from keras.applications import InceptionV3
from keras.preprocessing.image import ImageDataGenerator
from keras import models, layers
import matplotlib.pyplot as plt

# Instantiate the InceptionV3 model
conv_base = InceptionV3(weights='imagenet', # specifies which weight checkpoint to initialize the model from
                        include_top=False, # should not include the densely-connected classifier on top of the network.
                        input_shape=(150, 150, 3))


model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(5, activation='softmax'))

baseDir = "./datasets/flowers_split/"

model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["acc"])

trainDataGen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=50,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

testDataGen = ImageDataGenerator(rescale=1. / 255)

trainGenerator = trainDataGen.flow_from_directory(
    directory=os.path.join(baseDir, "train/"),
    target_size=(150, 150),
    batch_size=32,
    class_mode="categorical")

validationGenerator = testDataGen.flow_from_directory(
    directory=os.path.join(baseDir, "validation/"),
    target_size=(150, 150),
    batch_size=32,
    class_mode="categorical")

history = model.fit_generator(
    trainGenerator,
    steps_per_epoch=82,
    epochs=25,
    validation_data=validationGenerator,
    validation_steps=50)


pickle.dump(history, open('history_pretrained_aug.p', 'wb'))
model.save('flower_classification_model_pretrained_aug.h5')



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
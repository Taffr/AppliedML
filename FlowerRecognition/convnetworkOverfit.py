import os
from keras import layers, models, optimizers
import pickle
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras.datasets import mnist
# Change this to fit your machine
baseDir = "./datasets/flowers_split/"
model = models.Sequential()
# Add first convolution layer
model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(150, 150, 3))) # Change input shape to fit our images or maybe remove completly
model.add(layers.MaxPooling2D((2, 2))) # Add first downsizing layer
model.add(layers.Conv2D(64, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D((2, 2))) # Add second downsizing layer
model.add(layers.Conv2D(128, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D((2, 2))) # Add second downsizing layer
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation="relu"))
model.add(layers.Dense(5, activation="softmax"))
print(model.summary())

model.compile(loss="categorical_crossentropy", optimizer=optimizers.RMSprop(lr=1e-4), metrics=["acc"])


trainDataGen = ImageDataGenerator(rescale=1./255)
testDataGen = ImageDataGenerator(rescale=1./255)


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
    steps_per_epoch=81,
    epochs=10,
    validation_data=validationGenerator,
    validation_steps=50)


pickle.dump(history, open('history_overfit_with_relu.p', 'wb'))

model.save('flower_classification_model_with_relu.h5')


acc = history.history["acc"]
val_acc = history.history["val_acc"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]


loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()



print(acc)
print(val_acc)
print(loss)
print(val_loss)

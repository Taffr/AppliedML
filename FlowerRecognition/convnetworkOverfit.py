import os
from keras import layers, models, optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import mnist
# Change this to fit your machine
baseDir = "/home/simon/git/AppliedML/FlowerRecognition/datasets/flowers_split/"
model = models.Sequential()
# Add first convolution layer
model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(150, 150, 3))) # Change input shape to fit our images or maybe remove completly
model.add(layers.MaxPooling2D((2, 2))) # Add first downsizing layer
model.add(layers.Conv2D(64, (3, 3)))
model.add(layers.MaxPooling2D((2, 2))) # Add second downsizing layer
model.add(layers.Conv2D(128, (3, 3)))
model.add(layers.MaxPooling2D((2, 2))) # Add second downsizing layer
model.add(layers.Flatten())
model.add(layers.Dense(512, activation="relu"))
model.add(layers.Dense(5, activation="softmax"))
print(model.summary())

model.compile(loss="categorical_crossentropy", optimizer=optimizers.RMSprop(lr=1e-4), metrics=["acc"])


trainDataGen = ImageDataGenerator(rescale=1./255)
testDataGen = ImageDataGenerator(rescale=1./255)

trainGenerator = trainDataGen.flow_from_directory(
    directory=os.path.join(baseDir, "train/"),
    target_size=(150, 150),
    batch_size=20,
    class_mode="categorical")
validationGenerator = testDataGen.flow_from_directory(
    directory=os.path.join(baseDir, "validation/"),
    target_size=(150, 150),
    batch_size=20,
    class_mode="categorical")

history = model.fit_generator(
    trainGenerator,
    steps_per_epoch=100,
    epochs=30,
    validation_data=validationGenerator,
    validation_steps=50)

acc = history.history["acc"]
val_acc = history.history["val_acc"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]

print(acc)
print(val_acc)
print(loss)
print(val_loss)

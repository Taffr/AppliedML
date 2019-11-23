import os, pickle
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from sklearn import metrics

# model = load_model('flower_classification_model_overfit.h5')
# model.summary()
# # for layer in model.layers:
# #     weights = layer.get_weights()
# #     print(weights)
#
# testDataGen = ImageDataGenerator(rescale=1./255)
#
# testGenerator = testDataGen.flow_from_directory(
#     directory=os.path.join('./datasets/flowers_split/', "test/"),
#     target_size=(150, 150),
#     batch_size=1,
#     class_mode="categorical")
#
# loss, acc = model.evaluate_generator(testGenerator, steps=866, verbose=0)
# print(loss)
# print(acc)

history = pickle.load(open('history_aug.p', 'rb'))

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

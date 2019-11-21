import os
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from sklearn import metrics

model = load_model('flower_classification_model_aug.h5')
model.summary()
# for layer in model.layers:
#     weights = layer.get_weights()
#     print(weights)

testDataGen = ImageDataGenerator(rescale=1./255)

testGenerator = testDataGen.flow_from_directory(
    directory=os.path.join('./datasets/flowers_split/', "test/"),
    target_size=(150, 150),
    batch_size=1,
    class_mode="categorical")

loss, acc = model.evaluate_generator(testGenerator, steps= 866, verbose=0)
print(loss)
print(acc)

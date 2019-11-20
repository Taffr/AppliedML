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


predict = model.predict_generator(
    testGenerator,
    steps=866
)

print(predict)

result = []
for p in predict:
    max = -1
    max_index = -1
    for i in range(len(p)):
       if p[i] > max:
        max_index = i
        max = p[i]
    result.append(max_index)

print(result)

y_true = testGenerator.classes
print('** y_true **')
print(y_true)


print()
print(len(y_true))
print(len(result))
print(metrics.classification_report(y_true, result))
# print("Confusion matrix:\n%s"
#       % metrics.confusion_matrix(testLabels, predicted))


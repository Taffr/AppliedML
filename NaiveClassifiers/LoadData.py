from sklearn import datasets, metrics
import numpy as np

def modifyDigits(training, test):
    modifiedData = []
    for row in training:
        nArr = []
        for pixel in row:
            if pixel < 5:
                nArr.append(0)
            if 5 <= pixel < 10:
                nArr.append(1)
            if pixel >= 10:
                nArr.append(2)
        modifiedData.append(nArr)

    modifiedTestData = []
    for row in test:
        nArr = []
        for pixel in row:
            if pixel < 5:
                nArr.append(0)
            if 5 <= pixel < 10:
                nArr.append(1)
            if pixel >= 10:
                nArr.append(2)
        modifiedTestData.append(nArr)
    return np.asarray(modifiedData), np.asarray(modifiedTestData)

import NearestCentroidClassifier as NCC
digits = datasets.load_digits()
split = int(0.7 * len(digits.data))
trainingFeatures = digits.data[:split]
testFeatures = digits.data[split:]
trainingLabels = digits.target[:split]
testLabels = digits.target[split:]
trainingFeatures, testFeatures = modifyDigits(trainingFeatures, testFeatures)
print(trainingFeatures)
ncc = NCC.NCC()
ncc.fit(trainingFeatures, trainingLabels)
preds = ncc.predict(testFeatures)

print(metrics.classification_report(testLabels, preds))
print(metrics.confusion_matrix(testLabels, preds))




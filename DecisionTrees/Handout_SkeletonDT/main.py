import ToyData as td
import ID3

import numpy as np
from sklearn import tree, metrics, datasets


def main():
    #attributes, classes, data, target, data2, target2 = td.ToyData().get_data()
    digits = datasets.load_digits()
    split = int(0.7 * len(digits.data))
    trainingFeatures = digits.data[:split]
    testFeatures = digits.data[split:]
    trainingLabels = digits.target[:split]
    testLabels = digits.target[split:]
    id3 = ID3.ID3DecisionTreeClassifier()
    classes = list(range(0, 10))

    modifiedData = []
    for row in trainingFeatures:
        nArr = []
        for pixel in row:
            if pixel < 5:
                nArr.append("d")
            if 5 <= pixel < 10:
                nArr.append("g")
            if pixel >= 10:
                nArr.append("l")
        modifiedData.append(nArr)
    attributes = {}
    for i in range(0, 64):
        attributes["pixel" + str(i)] = ["d", "g", "l"]

    modifiedTestData = []
    for row in testFeatures:
        nArr = []
        for pixel in row:
            if pixel < 5:
                nArr.append("d")
            if 5 <= pixel < 10:
                nArr.append("g")
            if pixel >= 10:
                nArr.append("l")
        modifiedTestData.append(nArr)

    myTree = id3.fit(modifiedData, trainingLabels, attributes, classes)
    plot = id3.makeDotData()
    plot.render("testTree")
    predicted = id3.predict(modifiedTestData, myTree)
    print(predicted)
    print(metrics.classification_report(testLabels, predicted))
    print(metrics.confusion_matrix(testLabels, predicted))


if __name__ == "__main__": main()

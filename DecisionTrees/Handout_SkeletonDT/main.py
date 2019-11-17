import ToyData as td
import ID3

import numpy as np
from sklearn import tree, metrics, datasets


def main():
    attributes, classes, data, target, data2, target2 = td.ToyData().get_data()
    digits = datasets.load_digits()
    split = int(0.7 * len(digits.data))
    trainingFeatures = digits.data[:split]

    testFeatures = digits.data[split:]
    trainingLabels = digits.target[:split]
    testLabels = digits.target[split:]
    id3 = ID3.ID3DecisionTreeClassifier()
    classes = list(range(0, 10))
    attributes = {}
    #for index in range(64):
    #    attributes[index] = list(range(0, 17))

    attributes["dark"] = list(range(0, 5))
    attributes["grey"] = list(range(5, 10))
    attributes["light"] = list((range(10, 17)))

    myTree = id3.fit(trainingFeatures, trainingLabels, attributes, classes)
    plot = id3.makeDotData()
    plot.render("testTree")
    predicted = id3.predict(testFeatures, myTree)
    print(predicted)
    print(metrics.classification_report(testLabels, predicted))
    print(metrics.confusion_matrix(testLabels, predicted))
if __name__ == "__main__": main()
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
    #print(classes)
    print(trainingFeatures)
    #attributes =
    #myTree = id3.fit(trainingFeatures, trainingLabels, attributes, classes)
    #print(myTree)
    plot = id3.makeDotData()
    plot.render("testTree")
    #predicted = id3.predict(data2, myTree)
    #print(predicted)


if __name__ == "__main__": main()
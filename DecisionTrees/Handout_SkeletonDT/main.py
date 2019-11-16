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
    count = {}
    for l in trainingLabels:
        if l in count:
            count[l] += 1
        else:
            count[l] = 1

    print("**** COUNT ****")
    print(count)
    testLabels = digits.target[split:]
    id3 = ID3.ID3DecisionTreeClassifier()
    classes = list(range(0, 10))
    print(classes)
    print(trainingFeatures)
    attributes = {}
    # for row in range(8):
    for col in range(64):
        attributes[col] = list(range(17))
    print(attributes)

    myTree = id3.fit(trainingFeatures, trainingLabels, attributes, classes)
    print('\n ** myTree **')
    print(myTree)
    plot = id3.makeDotData()
    plot.render("testTree")
    predicted = id3.predict(testFeatures, myTree)


    print('\n PREDICTED')
    for p in predicted:
        print(p)

if __name__ == "__main__": main()
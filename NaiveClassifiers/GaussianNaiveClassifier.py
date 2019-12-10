from math import sqrt
from math import pi
from math import exp
import numpy as np

class GNC:
    def fit(self, features, target, featureValues, targetValues):
        self.dataset = zip(features, target)
        self.features = features
        self.targetValues = targetValues
        classSeperated = self.__seperateByClass(self.dataset, targetValues)
        self.classSummarized = dict()
        for classification, arrays in classSeperated.items():
            self.classSummarized[classification] = self.__summarizeDataset(arrays, featureValues)

    def __seperateByClass(self, dataset, targetValues):
        classSeperated = dict()
        for value in targetValues:
            classSeperated[value] = list()
        for row in dataset:
            classSeperated[row[1]].append(row[0])
        return classSeperated

    def __mean(self, l):
        return float(sum(l) / len(l))

    def __stdev(self, l):
        mean = self.__mean(l)
        variance = sum([(x - mean) ** 2 for x in l]) / float(len(l) - 1)
        return sqrt(variance)

    def __summarizeDataset(self, features, featureValues):
        colValues = dict()
        for i in range(len(featureValues.keys())):
            colValues[i] = list()

        for row in range(len(features)):
            for col in range(len(features[row])):
                colValues[col].append(features[row][col])

        summaries = [(self.__mean(col), self.__stdev(col), len(col)) for col in colValues.values()]
        return summaries

    def __gaussianProb(self, x, mean, std):
        epsilon = 0.0001
        expon = exp((-1 / 2) * ((x - mean) / (std + epsilon)) ** 2)
        return (1 / ((std + epsilon) * sqrt(2 * pi))) * expon

    def __probabilities(self, summaries, row):
        totalRows = len(self.features)
        probabilites = np.zeros(len(self.targetValues))
        for classification, summs in summaries.items():
            probabilites[classification] = float(summaries[classification][0][2] / totalRows) # summaries[classification][0][2] == count of class
            for i in range(len(summs)):
                mean, std, count = summs[i]
                probabilites[classification] *= self.__gaussianProb(row[i], mean, std)
        return probabilites

    def predict(self, toBePredicted):
        predictions = np.zeros(len(toBePredicted))
        counter = 0
        for row in toBePredicted:
            predictions[counter] = np.argmax(self.__probabilities(self.classSummarized, row))
            counter += 1
        return predictions

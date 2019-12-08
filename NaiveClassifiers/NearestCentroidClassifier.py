import sys
import copy
from collections import defaultdict

import numpy as np


class NCC:

    def fit(self, features, target):
        self.averages = dict()
        classCounter = dict()
        for classification in target:
            self.averages[classification] = 0
            classCounter[classification] = 0

        for row in list(zip(features, target)):
            for i in range(len(row[0])):
                try:
                    classCounter[row[1]] += 1
                    self.averages[row[1]][i] += row[0][i]
                except:
                    self.averages[row[1]] = np.zeros(len(row[0]))
                    self.averages[row[1]][i] = row[0][i]
                    print(row[0][i])
        for key in self.averages.keys():
            self.averages[key] = (1/classCounter[key]) * self.averages[key]
            print(self.averages)


    def predict(self, features):
        predicted = list(range((len(features))))
        for i in range(len(features)):
            predictedClass = 0
            lowest = sys.maxsize
            for meanArr in self.averages.items():
                distance = np.linalg.norm(features[i] - meanArr[1])
                if distance < lowest:
                    lowest = distance
                    predictedClass = meanArr[0]
            predicted[i] = predictedClass

        return predicted

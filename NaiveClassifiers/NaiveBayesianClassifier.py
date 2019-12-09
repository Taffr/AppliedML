import numpy as np


class NBC:

    def fit(self, features, target, featureValues, targetValues):
        dataset = list(zip(features, target))
        self.targetValues = targetValues
        # Make freq table for each feature
        freqTables = self.__frequencyTables(dataset, featureValues, targetValues)
        self.likelihoodtables = dict()
        self.classProbs = self.__classProbabilities(dataset, targetValues)
        for table in freqTables.items():
            key, table = self.__likelihoodTable(table, featureValues, targetValues)
            self.likelihoodtables[key] = table

    def __frequencyTables(self, dataset, featureValues, targetValues):
        freqTables = dict()
        # make a freq. table for each pixel
        for featureItem in featureValues.items():
            freqTables[featureItem[0]] = dict()
            for value in featureItem[1]:
                freqTables[featureItem[0]][value] = dict()
                for target in targetValues:
                    freqTables[featureItem[0]][value][target] = 0

        # loop through the dataset
        for row in range(len(dataset)):
            for col in range(len(dataset[row][0])):
                freqTables[col][dataset[row][0][col]][dataset[row][1]] += 1

        return freqTables

    def __classProbabilities(self, dataset, targetValues):
        classCounter = dict()
        for value in targetValues:
            classCounter[value] = 0

        for i in range(len(dataset)):
            classCounter[dataset[i][1]] += 1

        total = sum(classCounter.values())
        for classification in classCounter.keys():
            classCounter[classification] /= total

        return classCounter

    def __likelihoodTable(self, freqTable, featureValues, targetValues):
        likelihoodTable = dict()
        for value in featureValues[freqTable[0]]:
            likelihoodTable[value] = dict()
            for classification in targetValues:
                likelihoodTable[value][classification] = 0
        # freqTable[1] = value: {class: count, ...}, e.g. 3.0
        for value in freqTable[1].keys():
            valueTotal = sum(freqTable[1][value].values())
            for classCounter in freqTable[1][value].items():
                classification = classCounter[0]
                if valueTotal != 0:
                    prob = classCounter[1] / valueTotal
                    likelihoodTable[value][classification] = prob
                else:
                    likelihoodTable[value][classification] = 0

        return freqTable[0], likelihoodTable

    def predict(self, toBePredicted):
        epsilon = 0.001
        counter = 0
        predictions = np.zeros(len(toBePredicted))
        print()
        for featureVector in toBePredicted:
            pClasses = np.ones(len(self.targetValues))
            for fIndex, feature in enumerate(featureVector):
                for classProb in self.likelihoodtables[fIndex][feature].items():
                    pClasses[classProb[0]] *= (classProb[1] + epsilon)

            for i in range(len(pClasses)):
                pClasses[i] = pClasses[i] / (pClasses[i] + self.classProbs[i])
            print(pClasses)
            predictions[counter] = (np.argmax(pClasses))
            counter += 1
        return predictions

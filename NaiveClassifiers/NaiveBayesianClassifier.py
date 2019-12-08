class NBC:

    def fit(self, features, target, featureValues, targetValues):
        dataset = list(zip(features, target))
        # Make freq table for each feature
        freqTables = self.__frequencyTables(dataset, featureValues, targetValues)
        self.likelihoodtables = dict()
        self.classProbs = self.__classProbabilities(dataset, targetValues)
        for table in freqTables.items():
            key, table = self.__likelihoodTable(table, len(dataset))
            self.likelihoodtables[key] = table
        print(self.likelihoodtables)

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

    def __likelihoodTable(self, freqTable, total):
        likelihoodTable = dict()
        for value in freqTable[1].keys():
            likelihoodTable[value] = 0
            for val in freqTable[1][value].values():
                likelihoodTable[value] += val

        for key in likelihoodTable.keys():
            likelihoodTable[key] /= total

        return freqTable[0], likelihoodTable


    def predict(self, toBePredicted):
        print("TODO")
        return []
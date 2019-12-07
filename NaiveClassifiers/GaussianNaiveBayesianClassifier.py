class GaussianNaiveBayesianClassifier:
    def __init__(self, dataSetName):
        self.dataSetName = dataSetName

    def fit(self, features, target):
        dataset = list(zip(features, target))
        classSeperated = self.__seperateByClass__(dataset)

    def __seperateByClass__(self, dataset):
        # dataset[i][1] = class
        # dataset[i][0] = array

        # Inefficient lmao
        split = dict()
        for line in range(len(dataset)):
            split[str(dataset[line][1])] = list()

        for line in range(len(dataset)):
            split[str(dataset[line][1])].append(dataset[line][0])

        return split  # {class: list(list) ... }

    def __mean__(self, numbers):
        return sum(numbers) / len(numbers)

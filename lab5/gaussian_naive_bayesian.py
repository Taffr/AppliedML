import numpy as np
import math
from sklearn.metrics import classification_report, confusion_matrix


class GaussianNaiveBayesian:
    def __init__(self):
        self.y_pred = None
        self.mean_values = None
        self.variance_values = None
        self.class_probabilities = None


    def fit(self, X, Y):
        self.mean_values, class_count = self.__mean_values(X, Y)
        self.class_probabilities = self.__class_probabilities(Y, class_count)
        class_separation = self.__class_separation(X, Y)
        self.variance_values = self.__variance_values(class_separation, self.mean_values)


    def __variance_values(self, class_separation, mean_values):
        variance_values = {}
        for y, y_collection in class_separation.items():
            variance_values[y] = np.zeros(len(y_collection[0]))

            for values in y_collection:
                for value_i, value in enumerate(values):
                    variance_values[y][value_i] += (value - mean_values[y][value_i])**2

            variance_values[y] = np.divide(variance_values[y], len(y_collection))
        return variance_values

    def __mean_values(self, X, Y):
        class_count = {}
        mean_values = {}

        for i in range(len(X)):
            x = X[i]
            y = Y[i]

            if y in class_count:
                class_count[y] += 1
                for k in range(len(x)):
                    mean_values[y][k] += x[k]
            else:
                class_count[y] = 1
                mean_values[y] = np.array(x)

        for mean_key, mean_value in mean_values.items():
            mean_values[mean_key] = np.divide(mean_values[mean_key], class_count[mean_key])

        return mean_values, class_count

    def __class_separation(self, X, Y):
        separation = {}
        for i, x in enumerate(X):
            y = Y[i]
            if y in separation:
                separation[y].append(x)
            else:
                separation[y] = [x]

        return separation

    def __class_probabilities(self, Y, class_count):
        class_probabilities = {}
        for label in class_count:
            class_probabilities[label] = class_count[label] / len(Y)

        return class_probabilities

    def __gaussian_probability(self, x, mean, variation):
        epsilon = 0.01

        exp = math.exp(-(x - mean)**2 / (2 * (variation + epsilon)))
        probability = exp / math.sqrt(2 * math.pi * (variation + epsilon))
        #
        # print('x',x)
        # print('mean', mean)
        # print('variation', variation)
        # print('probability', probability)

        # 1 / sqrt(2 * pi * epsilon) = 1 / 0,02506628275 = 39,89

        return probability


    def predict(self, X):
        self.y_pred = []

        mean_values = self.mean_values
        variance_values = self.variance_values
        class_probabilities = self.class_probabilities

        # *** mean_values har för varje siffra, medelvärdet för varje pixel ***
        # print(len(mean_values)) # 10, en för varje siffra
        # print(len(mean_values[0])) # 64, en för varje pixel.

        # print(len(variance_values)) # 10, en för varje siffra
        # print(len(variance_values[0])) # 64, variansen för varje pixel för siffran 0

        for x_i, x in enumerate(X):
            probabilities = np.zeros(len(variance_values))
            for label_i, label in enumerate(variance_values):
                variance = variance_values[label]
                prob = 1
                for x_value_i, x_value in enumerate(x):
                    prob *= self.__gaussian_probability(x_value, mean_values[label][x_value_i], variance[x_value_i])
                prob *= class_probabilities[label]
                probabilities[label_i] = prob

            # print(max(probabilities))
            #normalize
            # sum_prob = sum(probabilities)
            # print(sum_prob)
            # if sum_prob > 0:
            #     np.divide(probabilities, sum_prob)
            self.y_pred.append(np.argmax(probabilities))

        return self.y_pred


    def create_reports(self, test_labels):
        clf_report = classification_report(test_labels, self.y_pred)
        conf_matrix = confusion_matrix(test_labels, self.y_pred)

        return clf_report, conf_matrix

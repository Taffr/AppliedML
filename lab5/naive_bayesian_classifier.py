import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


class NaiveBayesianClassifier:
    def __init__(self):
        self.class_probabilities = None
        self.likelihood_table = None
        self.y_pred = None

    def fit(self, X, Y):
        frequency_table = self.__frequency_table(X, Y)
        self.class_probabilities, class_count = self.__class_probabilities(Y)
        self.likelihood_table = self.__likelihood_table(frequency_table, class_count)

    def __class_probabilities(self, Y):
        class_count = {}
        class_probabilities = {}
        for y in Y:
            if y in class_count:
                class_count[y] += 1
            else:
                class_count[y] = 1

        n_total = len(Y)
        for label in class_count:
            class_probabilities[label] = class_count[label] / n_total

        return class_probabilities, class_count

    def __frequency_table(self, X, Y):
        frequency_table = {}

        for x_i, x in enumerate(X):
            y = Y[x_i]

            if y not in frequency_table:
                frequency_table[y] = {}
                for i in range(len(x)):
                    frequency_table[y][i] = {}

            for value_i, value in enumerate(x):
                if value in frequency_table[y][value_i]:
                    frequency_table[y][value_i][int(value)] += 1
                else:
                    frequency_table[y][value_i][int(value)] = 1

        # print(len(frequency_table)) # 10, en för varje siffra
        # print(len(frequency_table[0])) # 64, en för varje pixel
        # print(len(frequency_table[0][0])) # har antal för dem pixelvärden som förekommer
        # frequency_table[0][23][4] # antal gånger siffran 0, pixel 23 har värdet 4. Om inte 4:an finns, så är antalet gånger 0.

        return frequency_table

    def __likelihood_table(self, frequency_table, class_count):
        likelihood_table = {}

        for label, features in frequency_table.items():
            likelihood_table[label] = {}
            for feature, values in features.items():
                likelihood_table[label][feature] = {}
                for value, value_frequency in values.items():
                    likelihood_table[label][feature][value] = value_frequency / class_count[label]

        # samma format som frequency_table
        return likelihood_table

    def predict(self, X):
        self.y_pred = []
        epsilon = 0

        for x_i, x in enumerate(X):
            probabilities = np.zeros(len(self.likelihood_table))
            for label_i, label in enumerate(self.likelihood_table):
                likelihood = self.likelihood_table[label]
                prob = 1
                for x_value_i, x_value in enumerate(x):
                    prob *= likelihood[x_value_i].get(int(x_value), epsilon)
                prob *= self.class_probabilities[label]
                probabilities[label_i] = prob

            # normalize
            sum_prob = sum(probabilities)
            if sum_prob > 0:
                np.divide(probabilities, sum_prob)

            self.y_pred.append(np.argmax(probabilities))

        return self.y_pred

    def create_reports(self, Y):
        clf_report = classification_report(Y, self.y_pred)
        conf_matrix = confusion_matrix(Y, self.y_pred)
        return clf_report, conf_matrix

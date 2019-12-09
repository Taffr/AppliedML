import numpy as np
from sklearn import metrics

class NaiveBayesianClassifier:
    def __init__(self, dimension, n_features):
        self.dimension = dimension
        self.n_features = n_features


    def fit(self, train_features, train_labels):
        nbr_features = len(train_features)
        pixel_count_per_digit = {}

        self.class_probabilities = self.__class_probabilities(train_labels)

        for digit_i in range(len(train_features)):
            digit = train_features[digit_i]
            label = train_labels[digit_i]

            if label not in pixel_count_per_digit:
                pixel_count_per_digit[label] = np.zeros((self.dimension, self.n_features), dtype=int)

            for pixel_i in range(len(digit)):
                pixel_value = digit[pixel_i]
                pixel_count_per_digit[label][pixel_i][int(pixel_value)] += 1

        pixel_probabilities = {}
        digit_probabilities = np.zeros(10)
        for digit_i in range(10):
            pixel_probabilities[digit_i] = np.divide(pixel_count_per_digit[digit_i], self.class_probabilities[digit_i])
            digit_probabilities[digit_i] = self.class_probabilities[digit_i] / nbr_features

        self.pixel_probabilities = pixel_probabilities


    def __class_probabilities(self, train_labels):
        class_probabilities = {}
        for label in train_labels:
            if label in class_probabilities:
                class_probabilities[label] += 1
            else:
                class_probabilities[label] = 1

        n_total = len(train_labels)
        for label in class_probabilities:
            class_probabilities[label] /= n_total

        return class_probabilities


    def predict(self, test_features):
        y_pred = np.zeros(len(test_features), dtype=int)

        # print(pixel_probabilities[0])
        # print(len(pixel_probabilities)) # 10, en för varje siffra
        # print(len(pixel_probabilities[0])) # 17, en för varje pixelvärde
        # print(len(pixel_probabilities[0][0])) # 64, en för varje pixel
        # print(pixel_probabilities[0][25][13]) # sannolikheten att siffra 0, pixel 25 har värde 13


        for feature_i, feature in enumerate(test_features):
            probabilities = np.zeros(10)
            for digit_i in range(10):
                prob = 1
                for pixel_i in range(self.dimension):
                    feature_pixel = feature[pixel_i]
                    prob *= self.pixel_probabilities[digit_i][pixel_i][int(feature_pixel)]
                prob *= self.class_probabilities[digit_i]
                probabilities[digit_i] = prob

            y_pred[feature_i] = np.argmax(probabilities)

        self.y_pred = y_pred
        return y_pred




    def create_reports(self, test_labels):
        clf_report = metrics.classification_report(test_labels, self.y_pred)
        conf_matrix = metrics.confusion_matrix(test_labels, self.y_pred)
        return clf_report, conf_matrix

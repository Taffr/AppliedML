import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


class NearestCentroidClassifier:
    def __init__(self):
        self.y_pred = None
        self.mean_values = None

    def fit(self, X, Y):
        classification_count = {}
        self.mean_values = {}

        for i in range(len(X)):
            x = X[i]
            y = Y[i]

            if y in classification_count:
                classification_count[y] += 1
                for k in range(len(x)):
                    self.mean_values[y][k] += x[k]
            else:
                classification_count[y] = 1
                self.mean_values[y] = np.array(x)

        for mean_key, mean_value in self.mean_values.items():
            for i in range(len(mean_value)):
                self.mean_values[mean_key][i] /= classification_count[mean_key]


    def predict(self, X):
        self.y_pred = np.zeros(len(X))
        mean_values_keys = list(self.mean_values)

        for i in range(len(X)):
            x = np.array(X[i])
            distances = []
            for mean_value in self.mean_values.values():
                d = np.linalg.norm(x - mean_value)
                distances.append(d)
            self.y_pred[i] = mean_values_keys[np.argmin(distances)]

        return self.y_pred

    def create_reports(self, Y):
        clf_report = classification_report(Y, self.y_pred)
        conf_matrix = confusion_matrix(Y, self.y_pred)
        return clf_report, conf_matrix


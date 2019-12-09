import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


class NearestCentroidClassifier:
    def fit(self, train_features, train_labels):
        classification_count = {}
        mean_values = {}

        for i in range(len(train_features)):
            feature = train_features[i]
            label = train_labels[i]

            if label in classification_count:
                classification_count[label] += 1
                for k in range(len(feature)):
                    mean_values[label][k] += feature[k]
            else:
                classification_count[label] = 1
                mean_values[label] = np.array(feature)

        for mean_key, mean_value in mean_values.items():
            for i in range(len(mean_value)):
                mean_values[mean_key][i] /= classification_count[mean_key]

        self.mean_values = mean_values


    def predict(self, test_features):
        y_pred = np.zeros(len(test_features))
        mean_values_keys = list(self.mean_values)

        for i in range(len(test_features)):
            feature = np.array(test_features[i])
            distances = []
            for mean_value in self.mean_values.values():
                d = np.linalg.norm(feature - mean_value)
                distances.append(d)
            y_pred[i] = mean_values_keys[np.argmin(distances)]

        self.y_pred = y_pred
        return y_pred

    def create_reports(self, test_labels):
        clf_report = classification_report(test_labels, self.y_pred)
        conf_matrix = confusion_matrix(test_labels, self.y_pred)
        return clf_report, conf_matrix


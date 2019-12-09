import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

class NearestCentroidClassifier:
    def fit(self, train_features, train_labels):
        counts = {}
        mean_values = {}

        for i in range(len(train_features)):
            pixel_values = train_features[i]
            label = train_labels[i]

            if label in counts:
                counts[label] += 1

                for k in range(len(pixel_values)):
                    mean_values[label][k] += pixel_values[k]
            else:
                counts[label] = 1
                mean_values[label] = np.array(pixel_values)

        for digit in mean_values:
            for i in range(len(mean_values[digit])):
                mean_values[digit][i] /= counts[digit]

        self.mean_values = mean_values


    def predict(self, test_features):
        y_pred = np.zeros(len(test_features))
        mean_values_keys = list(self.mean_values)

        for i in range(len(test_features)):
            digit = np.array(test_features[i])
            distances = []
            for l in self.mean_values:
                d = np.linalg.norm(digit - self.mean_values[l])
                distances.append(d)
            y_pred[i] = mean_values_keys[np.argmin(distances)]

        self.y_pred = y_pred
        return y_pred

    def create_reports(self, test_labels):
        clf_report = classification_report(test_labels, self.y_pred)
        conf_matrix = confusion_matrix(test_labels, self.y_pred)
        return clf_report, conf_matrix


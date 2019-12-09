from sklearn import datasets

def digits_dataset():
    digits = datasets.load_digits()
    nbr_splits = int(0.7 * len(digits.data))
    train_features = digits.data[:nbr_splits]
    train_labels = digits.target[:nbr_splits]

    test_features = digits.data[nbr_splits:]
    test_labels = digits.target[nbr_splits:]

    return train_features, train_labels, test_features, test_labels

def reduce_digits_dataset():
    train_features, train_labels, test_features, test_labels = digits_dataset()

    train_features_reduced = []
    test_features_reduced = []

    for digit in train_features:
        reduced = []
        for pixel_value in digit:
            if pixel_value < 5:
                reduced.append(0)
            elif pixel_value < 10:
                reduced.append(1)
            else:
                reduced.append(2)
        train_features_reduced.append(reduced)


    for digit in test_features:
        reduced = []
        for pixel_value in digit:
            if pixel_value < 5:
                reduced.append(0)
            elif pixel_value < 10:
                reduced.append(1)
            else:
                reduced.append(2)
        test_features_reduced.append(reduced)

    return train_features_reduced, train_labels, test_features_reduced, test_labels


if __name__ == '__main__':
    train_features, train_labels, test_features, test_labels = digits_dataset()
    train_features_reduced, train_labels, test_features_reduced, test_labels = reduce_digits_dataset()

    print(train_features_reduced[0])
    print(train_features[0])

    print(train_labels)
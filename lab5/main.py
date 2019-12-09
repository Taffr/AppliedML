import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from nearest_centroid_classifier import NearestCentroidClassifier
from naive_bayesian_classifier import NaiveBayesianClassifier
import preprocess_digits_dataset
import MNIST
import Main_MNIST

def run_gaussian_nb():
    train_features, train_labels, test_features, test_labels = preprocess_digits_dataset.digits_dataset()
    gaussian_digits = GaussianNB()
    gaussian_digits.fit(train_features, train_labels)
    y_pred = gaussian_digits.predict(test_features)
    digits_clf_report = classification_report(test_labels, y_pred)
    digits_conf_matrix = confusion_matrix(test_labels, y_pred)

    train_features, train_labels, test_features, test_labels = preprocess_digits_dataset.reduce_digits_dataset()
    gaussian_digits_reduced = GaussianNB()
    gaussian_digits_reduced.fit(train_features, train_labels)
    y_pred = gaussian_digits_reduced.predict(test_features)
    reduced_digits_clf_report = classification_report(test_labels, y_pred)
    reduced_digits_conf_matrix = confusion_matrix(test_labels, y_pred)

    mnist_clf_report, mnist_conf_matrix = Main_MNIST.main()


    write_reports_to_file(
        'results/gaussian_nb reports.txt',
        digits_clf_report,
        digits_conf_matrix,
        reduced_digits_clf_report,
        reduced_digits_conf_matrix,
        mnist_clf_report,
        mnist_conf_matrix
    )

def run_ncc():
    train_features, train_labels, test_features, test_labels = preprocess_digits_dataset.digits_dataset()
    ncc_digits = NearestCentroidClassifier()
    ncc_digits.fit(train_features, train_labels)
    ncc_digits.predict(test_features)
    digits_clf_report, digits_conf_matrix = ncc_digits.create_reports(test_labels)

    ncc_reduced_digits = NearestCentroidClassifier()
    train_features, train_labels, test_features, test_labels = preprocess_digits_dataset.reduce_digits_dataset()
    ncc_reduced_digits.fit(train_features, train_labels)
    ncc_reduced_digits.predict(test_features)
    reduced_digits_clf_report, reduced_digits_conf_matrix = ncc_reduced_digits.create_reports(test_labels)


    ncc_mnist = NearestCentroidClassifier()
    mnist = MNIST.MNISTData('MNIST_Light/*/*.png')
    train_features, test_features, train_labels, test_labels = mnist.get_data()
    ncc_mnist.fit(train_features, train_labels)
    ncc_mnist.predict(test_features)
    mnist_clf_report, mnist_conf_matrix = ncc_mnist.create_reports(test_labels)

    write_reports_to_file(
        'results/nearest_centroid_reports.txt',
        digits_clf_report,
        digits_conf_matrix,
        reduced_digits_clf_report,
        reduced_digits_conf_matrix,
        mnist_clf_report,
        mnist_conf_matrix
    )

def run_nbc():
    train_features, train_labels, test_features, test_labels = preprocess_digits_dataset.digits_dataset()
    nbc_digits = NaiveBayesianClassifier(64, 17)
    nbc_digits.fit(train_features, train_labels)
    nbc_digits.predict(test_features)
    digits_clf_report, digits_conf_matrix = nbc_digits.create_reports(test_labels)

    train_features, train_labels, test_features, test_labels = preprocess_digits_dataset.reduce_digits_dataset()
    nbc_reduced_digits = NaiveBayesianClassifier(64, 3)
    nbc_reduced_digits.fit(train_features, train_labels)
    nbc_reduced_digits.predict(test_features)
    reduced_digits_clf_report, reduced_digits_conf_matrix = nbc_reduced_digits.create_reports(test_labels)

    write_reports_to_file(
        'results/naive_bayesian_reports.txt',
        digits_clf_report,
        digits_conf_matrix,
        reduced_digits_clf_report,
        reduced_digits_conf_matrix,
    )


def write_reports_to_file(path,
                          digits_clf_report,
                          digits_conf_matrix,
                          reduced_digits_clf_report,
                          reduced_digits_conf_matrix,
                          mnist_clf_report=None,
                          mnist_conf_matrix=None):
    f_out = open(path, 'w')
    f_out.write('********** Digits **********\n')
    f_out.write('Classification report\n')
    f_out.write(digits_clf_report)
    f_out.write('\nConfusion matrix\n')
    f_out.write(np.array2string(digits_conf_matrix, separator=', '))

    f_out.write('\n\n\n********** Reduced digits **********\n')
    f_out.write('Classification report\n')
    f_out.write(reduced_digits_clf_report)
    f_out.write('\nConfusion matrix\n')
    f_out.write(np.array2string(reduced_digits_conf_matrix, separator=', '))

    if mnist_clf_report is not None and mnist_conf_matrix is not None:
        f_out.write('\n\n\n********** MNIST_Light **********\n')
        f_out.write('Classification report\n')
        f_out.write(mnist_clf_report)
        f_out.write('\nConfusion matrix\n')
        f_out.write(np.array2string(mnist_conf_matrix, separator=', '))


if __name__ == '__main__':
    run_gaussian_nb()
    run_ncc()
    run_nbc()

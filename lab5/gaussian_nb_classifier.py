import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.naive_bayes import GaussianNB
import Main_MNIST
import preprocess_digits_dataset

def run_gaussian_nb():
    train_features_digits,\
    train_labels_digits,\
    test_features_digits,\
    test_labels_digits = preprocess_digits_dataset.digits_dataset()

    train_features_digits_reduced,\
    train_labels_digits_reduced,\
    test_features_digits_reduced,\
    test_labels_digits_reduced = preprocess_digits_dataset.reduce_digits_dataset()

    gaussian_digits = GaussianNB()
    gaussian_digits.fit(train_features_digits, train_labels_digits)
    y_pred_digits = gaussian_digits.predict(test_features_digits)

    gaussian_digits_reduced = GaussianNB()
    gaussian_digits_reduced.fit(train_features_digits_reduced, train_labels_digits_reduced)
    y_pred_digits_reduced = gaussian_digits_reduced.predict(test_features_digits_reduced)



    digits_clf_report = classification_report(test_labels_digits, y_pred_digits)
    digits_conf_matrix = confusion_matrix(test_labels_digits, y_pred_digits)

    reduced_digits_clf_report = classification_report(test_labels_digits_reduced, y_pred_digits_reduced)
    reduced_digits_conf_matrix = confusion_matrix(test_labels_digits_reduced, y_pred_digits_reduced)

    mnist_clf_report, mnist_conf_matrix = Main_MNIST.main()


    f_out = open('results/gaussian_nb reports_2.txt', 'w')
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

    f_out.write('\n\n\n********** MNIST_Light **********\n')
    f_out.write('Classification report\n')
    f_out.write(mnist_clf_report)
    f_out.write('\nConfusion matrix\n')
    f_out.write(np.array2string(mnist_conf_matrix, separator=', '))


if __name__ == '__main__':
    run_gaussian_nb()


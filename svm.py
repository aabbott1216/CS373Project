from sklearn.svm import SVC
import numpy
import pandas as pd


# Testing function on dummy datapoints
def run(heart):
    # Separating into independent X and dependent y variables
    X = heart[list(heart.columns[:13])]
    y = heart["target"]

    svm_classifier = SVC()
    svm_classifier = svm_classifier.fit(X, y)

    # Dummy datapoints for proof of testing
    test_data = [[103, 1, 3, 121, 233, 1, 0, 65, 0, 2.3, 0, 0, 1],
                 [18, 1, 2, 130, 100, 0, 1, 7, 0, 3.5, 0, 0, 2],
                 [54, 0, 1, 115, 200, 0, 0, 172, 0, 1.4, 2, 0, 2]]
    return svm_classifier.predict(test_data)


# Fits data to svm classifier model and returns the model
def run2(x_train, y_train):
    # Fits the data to a validation tree.
    # The minimal sample split parameter requires X values in a node before it will be split (can be modified).
    svm_classifier = SVC()
    svm_classifier = svm_classifier.fit(x_train, y_train)
    return svm_classifier
    # return tuple((svm_classifier.predict(x_test), y_test))


# Predicts classification of given test data using given model
def prediction(classifier, test):
    return classifier.predict(test)

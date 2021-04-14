from sklearn.svm import SVC
import numpy
import pandas as pd


def run(heart):
    # Separating into independent X and dependent y variables
    X = heart[list(heart.columns[:13])]
    y = heart["target"]

    # Fits the data to a validation tree.
    # The minimal sample split parameter requires X values in a node before it will be split (can be modified).
    svm_classifier = SVC()
    svm_classifier = svm_classifier.fit(X, y)

    # Dummy datapoints for proof of testing
    test_data = [[103, 1, 3, 121, 233, 1, 0, 65, 0, 2.3, 0, 0, 1],
                 [18, 1, 2, 130, 100, 0, 1, 7, 0, 3.5, 0, 0, 2],
                 [54, 0, 1, 115, 200, 0, 0, 172, 0, 1.4, 2, 0, 2]]
    return svm_classifier.predict(test_data)


if __name__ == "__main__":
    df = pd.read_csv("resources/heart_dataset.csv")
    print(run(df))

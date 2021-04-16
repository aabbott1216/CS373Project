import validationtree
import svm
import numpy as np

from svm import run2 as svm_run
from svm import prediction as svm_pred
from validationtree import run2 as tree_run
from validationtree import prediction as tree_pred


# Input: numpy matrix X of features, with n rows (samples), d columns (features)
# numpy vector y of scalar values, with n rows (samples), 1 column
# Output: numpy vector z of k rows, 1 column
def run(heart, alg):

    y = heart['target']
    X = heart.drop(labels='target', axis=1)
    n = y.shape[0]
    k = 15
    z = [0]*k

    for i in range(k):
        T = range(int(n*(float(i)/k)), int(n*(float(i)+1)/k))
        T_set = set(T)
        S = set(range(0, n))-T_set
        S_list = list(S)
        T_list = list(T)

        Xtrain = X[S_list[0]:S_list[-1]]
        ytrain = y[S_list[0]:S_list[-1]]
        Xtest = X[T_list[0]:T_list[-1]]
        ytest = y[T_list[0]:T_list[-1]]

        if alg == "svm":
            classifier = svm_run(Xtrain, ytrain)
            prediction = list(svm_pred(classifier, Xtest))
            ytest = list(ytest)
            for ii in range(len(prediction)):
                if prediction[ii] == ytest[ii]:
                    z[i] += 1

        elif alg == "tree":
            tree = validationtree.run2(Xtrain, ytrain)
            prediction = list(tree_pred(tree, Xtest))
            ytest = list(ytest)
            for ii in range(len(prediction)):
                if prediction[ii] == ytest[ii]:
                    z[i] += 1
        z[i] /= float(len(prediction))
    return z

# def run(heart):
#     from svm import run2
#     import numpy as np
#     y = heart['target']
#     X = heart[list(heart.columns[:13])]
#     n = len(y)
#     k = 15
#     z = [0]*k
#     for i in range(k):
    # T = range(int(n*(float(i)/k)), int(n*(float(i)+1)/k))
    # T_set = set(T)
    # S = set(range(0, n))-T_set
    # S_list = list(S)
    # T_list = list(T)
    # X_train = X[S_list[0]:S_list[-1]]
    # y_train = y[S_list[0]:S_list[-1]]
    # X_test = X[T_list[0]:T_list[-1]]
    # y_test = y[T_list[0]:T_list[-1]]
#         results = []
#         for t in T:
#             results.append(run2(X_train, y_train, X_test, y_test))
#     return results


# if __name__ == "__main__":
#     import pandas as pd
#     heart = pd.read_csv("resources/heart_dataset.csv")
#     print(run(heart))

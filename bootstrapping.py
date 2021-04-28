import validationtree
import svm
import numpy as np
import pandas as pd

from svm import run2 as svm_run
from svm import prediction as svm_pred
from validationtree import run2 as tree_run
from validationtree import prediction as tree_pred
from SensitivitySpecificity import run as get_sens_spec
from roccurve import run as roc_plot
from HyperVsAcc import run as acc_plot

# Bootstrapping implementation for both the SVM and decision tree algorithms.


def run(heart, alg):

    # Forms needed matrices.
    y = heart['target']
    X = heart.drop(labels='target', axis=1)
    n = y.shape[0]
    b = 30
    z_svm = [0]*b
    z_tree = [0]*b
    svm_hyperparam = [0.1, 0.2, 0.5, 0.7, 0.9, 1, 2,
                      3, 5, 10, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 225, 250]
    tree_hyperparam = [0, .05, .1, .15, .20, .25, .30, .35, .40, .45, .50]
    svm_df = pd.DataFrame(
        columns=['C', 'Accuracy', 'Sensitivity', 'Specificity', 'Fold'])
    tree_df = pd.DataFrame(
        columns=['gini', 'Accuracy', 'Sensitivity', 'Specificity', 'Fold'])

    # Runs the bootstrapping cross-validation.
    for i in range(b):

        # Determines random set.
        u = np.zeros(n)
        S = set()
        for j in range(n):
            k = np.random.randint(0, n, 1)[0]
            u[j] = k
            S.add(k)
        T = set(range(n))
        T -= S

        T_list = list(T)
        S_list = list(S)

        # Forms training and testing sets.
        Xtrain = X[S_list[0]:S_list[-1]]
        ytrain = y[S_list[0]:S_list[-1]]
        Xtest = X[T_list[0]:T_list[-1]]
        ytest = y[T_list[0]:T_list[-1]]

        # Bootstrapping for SVM.
        if "svm" in alg:

            # LEGACY CODE
            # classifier = svm_run(X_train,y_train)
            # for t in T:
            #     if y[t] != svm_pred(classifier, X[t:t+1,:].T):
            #         z[i] = z[i]+1

            # Store hyperparameter accuracies.
            hyper_accs = [0]*len(svm_hyperparam)

            # For each hyperparameter, make a new model with that hyperparameter value and get the accuracy, sensitivity, and specificity to store in svm_df.
            # Essentially copied from the kfold.py file.
            for iii, hyperparam in enumerate(svm_hyperparam):
                classifier_svm = svm_run(Xtrain, ytrain, hyperparam)
                prediction_svm = list(svm_pred(classifier_svm, Xtest))
                ytest = list(ytest)
                pred_length = len(prediction_svm)
                sens, spec = get_sens_spec(ytest, prediction_svm)
                for ii in range(len(prediction_svm)):
                    if prediction_svm[ii] == ytest[ii]:
                        hyper_accs[iii] += 1.0
                temp_df = pd.DataFrame([[hyperparam, hyper_accs[iii]/pred_length, sens, spec, i]],
                                       columns=['C', 'Accuracy', 'Sensitivity', 'Specificity', 'Fold'])
                svm_df = svm_df.append(temp_df)

        # Bootstrapping for decision tree.
        if "tree" in alg:

            # LEGACY CODE
            # tree = tree_run(X_train,y_train)
            # for t in T:
            #     if y[t] != tree_pred(tree, X[t:t+1,:].T):
            #         z[i] = z[i]+1

            # Store hyperparameter accuracies
            hyper_accs_tree = [0]*len(tree_hyperparam)

            for iii, hyperparam in enumerate(tree_hyperparam):
                tree = tree_run(Xtrain, ytrain, hyperparam)
                prediction_tree = list(tree_pred(tree, Xtest))
                ytest = list(ytest)
                pred_length = len(prediction_tree)
                sens, spec = get_sens_spec(ytest, prediction_tree)

                for ii in range(len(prediction_tree)):
                    if prediction_tree[ii] == ytest[ii]:
                        hyper_accs_tree[iii] += 1
                temp_df = pd.DataFrame([[hyperparam, hyper_accs_tree[iii]/pred_length, sens, spec, i]],
                                       columns=['gini', 'Accuracy', 'Sensitivity', 'Specificity', 'Fold'])
                tree_df = tree_df.append(temp_df)

        # z[i] = z[i]/len(T)
        # z = z.reshape(z.shape[0],1)
        # z_svm[i] /= pred_length
        # z_tree[i] /= pred_length

    # ROC plot data
    roc_plot(svm_df['Sensitivity'], svm_df['Specificity'],
             tree_df['Sensitivity'], tree_df['Specificity'])
    # Accuracy vs hyperparam plot
    acc_plot(svm_df['C'], svm_df['Accuracy'],
             tree_df['gini'], tree_df['Accuracy'])

    # return z
    return (np.mean(z_svm), np.mean(z_tree))

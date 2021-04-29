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

    # Separate data into X features and y label
    y = heart['target']
    X = heart.drop(labels='target', axis=1)

    # Initialize bootstrap rounds, hyperparameters, and dataframe to store values
    n = y.shape[0]
    b = 30
    svm_hyperparam = [0.1, 0.2, 0.5, 0.7, 1, 5, 10,
                      20, 40, 60, 80, 100, 120, 140, 160, 180]
    tree_hyperparam = [0, 0.01, 0.03, .05, 0.75, .1,
                       0.125, .15, 0.175, .20, .25, .30, .35, .40, .45, .50]
    svm_df = pd.DataFrame(
        columns=['C', 'Accuracy', 'Sensitivity', 'Specificity', 'Round'])
    tree_df = pd.DataFrame(
        columns=['gini', 'Accuracy', 'Sensitivity', 'Specificity', 'Round'])

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

                # Compute accuracy and store information in dataframe
                for ii in range(len(prediction_svm)):
                    if prediction_svm[ii] == ytest[ii]:
                        hyper_accs[iii] += 1.0
                temp_df = pd.DataFrame([[hyperparam, hyper_accs[iii]/pred_length, sens, spec, i]],
                                       columns=['C', 'Accuracy', 'Sensitivity', 'Specificity', 'Round'])
                svm_df = svm_df.append(temp_df)

        # Bootstrapping for decision tree.
        if "tree" in alg:
            # Store hyperparameter accuracies
            hyper_accs_tree = [0]*len(tree_hyperparam)

            # For each hyperparameter, make a new model with that hyperparameter value and get the accuracy, sensitivity, and specificity to store in tree_df.
            # Essentially copied from the kfold.py file.
            for iii, hyperparam in enumerate(tree_hyperparam):
                tree = tree_run(Xtrain, ytrain, hyperparam)
                prediction_tree = list(tree_pred(tree, Xtest))
                ytest = list(ytest)
                pred_length = len(prediction_tree)
                sens, spec = get_sens_spec(ytest, prediction_tree)

                # Compute accuracy and store information in dataframe
                for ii in range(len(prediction_tree)):
                    if prediction_tree[ii] == ytest[ii]:
                        hyper_accs_tree[iii] += 1
                temp_df = pd.DataFrame([[hyperparam, hyper_accs_tree[iii]/pred_length, sens, spec, i]],
                                       columns=['gini', 'Accuracy', 'Sensitivity', 'Specificity', 'Round'])
                tree_df = tree_df.append(temp_df)

    # ROC plot data
    roc_plot(svm_df['Sensitivity'], svm_df['Specificity'],
             tree_df['Sensitivity'], tree_df['Specificity'], "boot")

    # Calculate mean Accuracy, Sensitivity, and Specificity for each C value across all samples
    mean_svm_df = (svm_df.groupby('C').mean()).reset_index()
    mean_tree_df = (tree_df.groupby('gini').mean()).reset_index()

    # Accuracy vs hyperparam plot
    acc_plot(mean_svm_df['C'], mean_svm_df['Accuracy'],
             mean_tree_df['gini'], mean_tree_df['Accuracy'], "boot")

    # return average accuracy, sensitivity, specificity across all 30 rounds
    return [mean_svm_df, mean_tree_df]

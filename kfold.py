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
# Run kfold cross validation on dataset to return average accuracy across all folds


def run(heart, alg):
    y = heart['target']
    X = heart.drop(labels='target', axis=1)
    n = y.shape[0]
    k = 15
    z_svm = [0]*k
    z_tree = [0]*k
    svm_hyperparam = [0.1, 0.2, 0.5, 0.7, 0.9, 1, 2,
                      3, 5, 10, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 225, 250]
    tree_hyperparam = [0, .05, .1, .15, .20, .25, .30, .35, .40, .45, .50]
    svm_df = pd.DataFrame(
        columns=['C', 'Accuracy', 'Sensitivity', 'Specificity', 'Fold'])
    tree_df = pd.DataFrame(
        columns=['gini', 'Accuracy', 'Sensitivity', 'Specificity', 'Fold'])

    for i in range(k):
        T = range(int(n*(float(i)/k)), int(n*(float(i)+1)/k))
        T_set = set(T)
        S = set(range(0, n))-T_set
        S_list = list(S)
        T_list = list(T)
        pred_length = 0

        Xtrain = X[S_list[0]:S_list[-1]]
        ytrain = y[S_list[0]:S_list[-1]]
        Xtest = X[T_list[0]:T_list[-1]]
        ytest = y[T_list[0]:T_list[-1]]

        if "svm" in alg:
            # Store hyperparameter accuracies
            hyper_accs_svm = [0]*len(svm_hyperparam)

            # For each hyperparameter, make a new model with that hyperparameter value and get the accuracy, sensitivity, and specificity to store in svm_df
            for iii, hyperparam in enumerate(svm_hyperparam):
                classifier_svm = svm_run(Xtrain, ytrain, hyperparam)
                prediction_svm = list(svm_pred(classifier_svm, Xtest))
                ytest = list(ytest)
                pred_length = len(prediction_svm)
                sens, spec = get_sens_spec(ytest, prediction_svm)

                for ii in range(len(prediction_svm)):
                    if prediction_svm[ii] == ytest[ii]:
                        hyper_accs_svm[iii] += 1.0
                temp_df = pd.DataFrame([[hyperparam, hyper_accs_svm[iii]/pred_length, sens, spec, i]], columns=[
                                       'C', 'Accuracy', 'Sensitivity', 'Specificity', 'Fold'])
                svm_df = svm_df.append(
                    temp_df)

        if "tree" in alg:
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
                temp_df = pd.DataFrame([[hyperparam, hyper_accs_tree[iii]/pred_length, sens, spec, i]], columns=[
                    'Gini', 'Accuracy', 'Sensitivity', 'Specificity', 'Fold'])
                tree_df = tree_df.append(
                    temp_df)

    # ROC plot data
    roc_plot(svm_df['Sensitivity'], svm_df['Specificity'],
             tree_df['Sensitivity'], tree_df['Specificity'])
    # Accuracy vs hyperparam plot
    acc_plot(svm_df['C'], svm_df['Accuracy'], 
             tree_df['Gini'], tree_df['Accuracy'])

    return (np.mean(z_svm), np.mean(z_tree))

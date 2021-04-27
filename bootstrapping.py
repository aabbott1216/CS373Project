import validationtree
import svm
import numpy as np

from svm import run2 as svm_run
from svm import prediction as svm_pred
from validationtree import run2 as tree_run
from validationtree import prediction as tree_pred
from SensitivitySpecificity import run as get_sens_spec

# Bootstrapping implementation for both the SVM and decision tree algorithms.
def run(heart, alg):

    # Forms needed matrices.
    y = heart['target'].to_numpy
    X = heart.drop(labels='target',axis=1).to_numpy
    n = len(y)
    b = 30
    z_svm = [0]*b
    z_tree = [0]*b
    svm_hyperparam = range(1, 30, 1)
    tree_hyperparam = []
    svm_df = pd.DataFrame(columns=['C', 'Accuracy', 'Sensitivity', 'Specificity', 'Fold'])

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

        # Forms training sets. 
        X_train = np.matrix(X[int(u[0])])
        y_train = np.matrix(y[int(u[0])])
        for j in range(1, n):
            X_train = np.vstack([X_train, X[int(u[j])]])
            y_train = np.vstack([y_train, y[int(u[j])]])

        # Forms testing sets.
        T_list = list(T)
        Xtest = X[T_list[0]:T_list[-1]]
        ytest = y[T_list[0]:T_list[-1]]
            
        # Bootstrapping for SVM.
        if alg=="svm":

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
        elif alg=="tree":
            tree = tree_run(X_train,y_train)
            for t in T:
                if y[t] != tree_pred(tree, X[t:t+1,:].T):
                    z[i] = z[i]+1

        # z[i] = z[i]/len(T)
        # z = z.reshape(z.shape[0],1)
        z_svm[i] /= pred_length
        z_tree[i] /= pred_length

    # ROC plot data
    # temp_data = svm_df.loc[svm_df["Fold"] == 0]
    # roc_plot(temp_data['Sensitivity'], temp_data['Specificity'])
        
    # return z
    return (np.mean(z_svm), np.mean(z_tree))

import numpy as np

from svm import run2 as svm_run
from svm import prediction as svm_pred
from validationtree import run2 as tree_run
from validationtree import prediction as tree_pred

# Bootstrapping implementation for both the SVM and decision tree algorithms.
def run(heart, alg):

    # Forms needed matrices.
    y = heart['target'].to_numpy
    X = heart.drop(labels='target',axis=1).to_numpy
    n = len(y)
    z = np.zeros(30)

    # Runs the bootstrapping cross-validation.
    for i in range(30):

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
            
        # Bootstrapping for SVM.
        if alg=="svm":
            classifier = svm_run(X_train,y_train)
            for t in T:
                if y[t] != svm_pred(classifier, X[t:t+1,:].T):
                    z[i] = z[i]+1

        # Bootstrapping for decision tree.            
        elif alg=="tree":
            tree = tree_run(X_train,y_train)
            for t in T:
                if y[t] != tree_pred(tree, X[t:t+1,:].T):
                    z[i] = z[i]+1

        z[i] = z[i]/len(T)
        z = z.reshape(z.shape[0],1)
        
    return z

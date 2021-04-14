import validationtree
import svm
import numpy as np

# Input: numpy matrix X of features, with n rows (samples), d columns (features)
# numpy vector y of scalar values, with n rows (samples), 1 column
# Output: numpy vector z of k rows, 1 column
def run(heart, alg):
    
    y = heart['target'].to_numpy
    X = heart.drop(labels='target',axis=1).to_numpy
    n = y.shape[0]
    z = np.zeros(15)

    for i in range(15):
        T = range(int(n*(float(i)/15)), int(n*(float(i)+1)/(15)))
        S = set(range(n)) - T
        T = list(T)
        S = list(S)
        Xtrain = X[np.array(S)]
        Ytrain = y[np.array(S)]
        
        if alg=="svm":
            classifier = svm.run(Xtrain,Ytrain)
            for t in T:
                if y[t] != svm.predict(classifier, X[t:t+1,:].T):
                    z[i] = z[i]+1
                    
        elif alg=="tree":
            tree = validationtree.run(Xtrain,Ytrain)
            for t in T:
                if y[t] != validationtree.predict(tree, X[t:t+1,:].T):
                    z[i] = z[i]+1

        z[i] = z[i]/len(T)
        z = z.reshape(z.shape[0],1)
        
    return z


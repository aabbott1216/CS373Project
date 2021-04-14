import numpy as np
import validationtree
import svm

# Input: number of bootstraps B
# numpy matrix X of features, with n rows (samples), d columns (features)
# numpy vector y of scalar values, with n rows (samples), 1 column
# Output: numpy vector z of B rows, 1 column
def run(heart, alg):
    y = heart['target'].to_numpy
    X = heart.drop(labels='target',axis=1).to_numpy
    n = len(y)
    z = np.zeros(30)
    for i in range(30):
        u = np.zeros(n)
        S = set()
        for j in range(n):
            k = np.random.randint(0, n, 1)[0]
            u[j] = k
            S.add(k)
        T = set(range(n))
        T -= S
        X_train = np.matrix(X[int(u[0])])
        y_train = np.matrix(y[int(u[0])])
        for j in range(1, n):
            X_train = np.vstack([X_train, X[int(u[j])]])
            y_train = np.vstack([y_train, y[int(u[j])]])
            
        if alg=="svm":
            classifier = svm.run(X_train,y_train)
            for t in T:
                if y[t] != svm.predict(classifier, X[t:t+1,:].T):
                    z[i] = z[i]+1
                    
        elif alg=="tree":
            tree = validationtree.run2(X_train,y_train)
            for t in T:
                if y[t] != validationtree.predict(tree, X[t:t+1,:].T):
                    z[i] = z[i]+1

        z[i] = z[i]/len(T)
        z = z.reshape(z.shape[0],1)
        
    return z

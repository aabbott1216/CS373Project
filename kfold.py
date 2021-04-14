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
            tree = validationtree.run2(Xtrain, Ytrain)
            for t in T:
                if y[t] != validationtree.predict(tree, X[t:t+1,:].T):
                    z[i] = z[i]+1

        z[i] = z[i]/len(T)
        z = z.reshape(z.shape[0],1)
        
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
#         T = range(int(n*(float(i)/k)), int(n*(float(i)+1)/k))
#         T_set = set(T)
#         S = set(range(0, n))-T_set
#         S_list = list(S)
#         T_list = list(T)
#         X_train = X[S_list[0]:S_list[-1]]
#         y_train = y[S_list[0]:S_list[-1]]
#         X_test = X[T_list[0]:T_list[-1]]
#         y_test = y[T_list[0]:T_list[-1]]
#         results = []
#         for t in T:
#             results.append(run2(X_train, y_train, X_test, y_test))
#     return results


# if __name__ == "__main__":
#     import pandas as pd
#     heart = pd.read_csv("resources/heart_dataset.csv")
#     print(run(heart))

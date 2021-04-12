# Input: number of folds k
# numpy matrix X of features, with n rows (samples), d columns (features)
# numpy vector y of scalar values, with n rows (samples), 1 column
# Output: numpy vector z of k rows, 1 column
def run(k, X, y):
    import probclearn
    import probcpredict
    import numpy as np
    n = len(y)
    z = [0]*k
    for i in range(k):
        T = range(int(n*(float(i)/k)), int(n*(float(i)+1)/(k)))
        T_set = set(T)
        S = set(range(0, n))-T_set
        S_list = list(S)

        X_train = np.matrix(X[S_list[0]])
        y_train = np.matrix(y[S_list[0]])
        for j in range(1, len(S_list)):
            X_train = np.vstack([X_train, X[S_list[j]]])
            y_train = np.vstack([y_train, y[S_list[j]]])
        q, mu_pos, mu_neg, var_pos, var_neg = probclearn.run(
            np.asarray(X_train), np.asarray(y_train))
        for t in T:
            if y[t] != probcpredict.run(q, mu_pos, mu_neg, var_pos, var_neg, np.resize(X[t], (len(X[t]), 1))):
                z[i] += 1
        z[i] /= len(T)
    return np.reshape(z, (len(z), 1))

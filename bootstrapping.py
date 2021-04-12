# Input: number of bootstraps B
# numpy matrix X of features, with n rows (samples), d columns (features)
# numpy vector y of scalar values, with n rows (samples), 1 column
# Output: numpy vector z of B rows, 1 column
def run(B, X, y):
    import numpy as np
    import probclearn
    import probcpredict
    n = len(y)
    z = [0]*B
    for i in range(B):
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
        q, mu_pos, mu_neg, var_pos, var_neg = probclearn.run(
            np.asarray(X_train), np.asarray(y_train))
        for t in T:
            if y[t] != probcpredict.run(q, mu_pos, mu_neg, var_pos, var_neg, np.resize(X[t], (len(X[t]), 1))):
                z[i] += 1
        z[i] /= len(T)
    return np.reshape(z, (len(z), 1))

from sklearn.svm import SVC
import numpy
def SVC(df):
    
    reg = svm.SVC()
    reg = reg.fit(X_train, Y_train)
    results(X_train, Y_train, X_test, Y_test,
            reg, "Support Vector Classification")

import numpy as np
import pandas as pd
import sklearn.tree

def run(heart):

    # Gets list of the columns that represent features.
    feature_list = list(heart.columns[:12])

    # Determines the X matrix (feature matrix)
    y = heart["target"]
    X = heart[feature_list]
    
    # Fits the data to a validation tree. 
    # The minimal sample split parameter requires X values in a node before it will be split (can be modified).
    tree = sklearn.tree.DecisionTreeClassifier(min_samples_split=10)
    tree.fit(X, y)

    print(tree)
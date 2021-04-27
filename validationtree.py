import numpy as np
import pandas as pd
import sklearn.tree
import subprocess


# Visualizes the data using graphviz. May not work if graphviz is not installed.
def show_tree(tree, features):

    with open("tree.dot", 'w') as f:
        sklearn.tree.export_graphviz(tree, out_file=f, feature_names=features)

    try:
        command = ["dot", "-Tpng", "tree.dot", "-o", "tree.png"]
        subprocess.check_call(command)
    except:
        print("An error occured translating DOT data into PNG. Use an online converter to see the tree.")

# # # #
# # # # # # # # # # End function


# Fits the data into a decision tree structure.
def run(heart):

    # Gets list of the columns that represent features.
    feature_list = list(heart.columns[:13])

    # Determines the X matrix (feature matrix)
    y = heart["target"]
    X = heart[feature_list]

    # Fits the data to a validation tree.
    # The minimal sample split parameter requires X values in a node before it will be split (can be modified).
    tree = sklearn.tree.DecisionTreeClassifier(criterion="gini", min_samples_split=10)
    tree.fit(X, y)

    # Runs the visualization before returning the current tree.
    show_tree(tree, feature_list)
    # return tree

    # Sample data prediction.
    test_data = [[103, 1, 3, 121, 233, 1, 0, 65, 0, 2.3, 0, 0, 1],
                 [18, 1, 2, 130, 100, 0, 1, 7, 0, 3.5, 0, 0, 2],
                 [54, 0, 1, 115, 200, 0, 0, 172, 0, 1.4, 2, 0, 2]]
    return tree.predict(test_data)

# # # #
# # # # # # # # # # End function


# Fits data into a decision tree structure, but doesn't use the whole dataframe.
def run2(X, y, gini_split):

    # Grabs the labels from the columns in X.
    feature_list = list(X.columns)

    # Fits the data to a validation tree.
    # The minimal sample split (hyper)parameter requires X values in a node before it will be split. It could be modified.
    # Minimum impurity decrease is the hyperparameter we are tuning. A higher value will cause less division. Default is 0.0 (float).
    tree = sklearn.tree.DecisionTreeClassifier(criterion="gini", min_samples_split=10, min_impurity_decrease=gini_split)
    tree.fit(X, y)

    # Runs the visualization before returning the current tree.
    show_tree(tree, feature_list)
    return tree

# # # #
# # # # # # # # # # End function


# Makes prediction for test data based on a passed tree.
def prediction(tree, test):
    
    # Returns prediction.
    return tree.predict(test)

# # # #
# # # # # # # # # # End function


if __name__ == "__main__":
    df = pd.read_csv("resources/heart_dataset.csv")
    print(run(df))


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Note: Code adapted from http://chrisstrelioff.ws/sandbox/2015/06/08/decision_trees_in_python_with_scikit_learn_and_pandas.html  #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

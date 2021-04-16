import numpy as np
import pandas as pd
from svm import run as run_svm
from validationtree import run as run_tree
from kfold import run as k_fold_validation


def pre_run():
    # Reads in CSV file using Pandas.
    heart = pd.read_csv('resources/heart_dataset.csv', delimiter=',')

    # Randomizes the original CSV file so that positive and negative labels are not grouped together.
    indexlist = np.random.choice(303, 303, replace=False)
    heartrandom = heart.copy()
    for i in range(303):
        heartrandom.iloc[i, :] = heart.iloc[indexlist[i], :]

    # Remove rows with null values for thal and ca
    heartrandom = heartrandom.drop(
        index=heartrandom[heartrandom['thal'] == 0].index)
    heartrandom = heartrandom.drop(
        index=heartrandom[heartrandom['ca'] == 4].index)

    # Replace thal values to match real world meaning of dataset
    heartrandom.loc[heartrandom.thal == 3, 'thal'] = 7
    heartrandom.loc[heartrandom.thal == 2, 'thal'] = 3
    heartrandom.loc[heartrandom.thal == 1, 'thal'] = 6

    print("Validation tree accuracy over 15 folds: ",
          k_fold_validation(heartrandom, "tree"))
    print("SVM accuracy over 15 folds: ",
          k_fold_validation(heartrandom, "svm"))


if __name__ == "__main__":
    df = pd.read_csv("resources/heart_dataset.csv")
    print(pre_run())

import numpy as np
import pandas as pd

# Reads in CSV file using Pandas.
heart = pd.read_csv('resources/heart_dataset.csv', delimiter=',')

# Randomizes the original CSV file so that positive and negative labels are not grouped together.
indexlist = np.random.choice(303,303,replace = False)
heartrandom = heart.copy()
for i in range(303):
    heartrandom.iloc[i,:] = heart.iloc[indexlist[i],:]
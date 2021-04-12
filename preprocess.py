import numpy as np
import pandas as pd

heart = pd.read_csv('heart.csv', delimiter=',')
indexlist = np.random.choice(303,303,replace = False)
heartrandom = heart.copy()
for i in range(303):
    heartrandom.iloc[i,:] = heart.iloc[indexlist[i],:]


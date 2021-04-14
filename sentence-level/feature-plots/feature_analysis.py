import os
import pandas as pd
import numpy as np

for file in os.listdir("./"):
    if file.endswith(".csv"):
        print(file)
        values = pd.read_csv(file, delimiter=" ", header=None)
        values = values.drop_duplicates()
        mean = np.mean(values[1])
        std = np.std(values[1])
        print(mean), np.std(std)
        outliers = values[values[1] > mean+std]
        outliers = outliers.append(values[values[1] < mean-std])
        print(outliers)
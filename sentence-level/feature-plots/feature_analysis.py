import os
import pandas as pd
import numpy as np

for file in os.listdir("./"):
    if file.endswith(".csv"):
        #print(file)
        if "zucoAll" in file:
            values = pd.read_csv(file, delimiter=" ", header=None)
            values = values.drop_duplicates()
            mean = np.mean(values[1])
            std = np.std(values[1])
            #print(mean, std)
            # todo: how many stds?
            outliers = values[values[1] > mean+(2*std)]
            outliers = outliers.append(values[values[1] < mean-(2*std)])
            #print(outliers)
            if not outliers.empty:
                print(file, mean, std, ",".join(outliers.iloc[:, 0].tolist()))
            else:
                print(file, mean, std, "None")
            #print("------------")

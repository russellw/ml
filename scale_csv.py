import argparse

from sklearn import preprocessing
import numpy as np
import pandas as pd

# args
parser = argparse.ArgumentParser()
parser.add_argument("filename", help="CSV file")
args = parser.parse_args()

# data
df = pd.read_csv(args.filename)
print(df)
print()

# numpy array
ar = np.array(df, dtype=np.float32)

# scale values to 0-1 range
ar = preprocessing.MinMaxScaler().fit_transform(ar)

# data frame
df = pd.DataFrame(data=ar, columns=df.columns)
print(df)

# write
df.to_csv(args.filename, index=False)

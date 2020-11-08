from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd


# data
df = pd.read_csv("test.csv")
print(df)
print()

# separate the output column
y_name = df.columns[-1]
y_df = df[y_name]
X_df = df.drop(y_name, axis=1)

# numpy arrays
X_ar = np.array(X_df, dtype=np.float32)
y_ar = np.array(y_df, dtype=np.float32)

# model
rf = RandomForestRegressor()

# train
print("training")
rf.fit(X_ar, y_ar)

predicted = rf.predict(X_ar)
errors = abs(predicted - y_ar)
print("mean squared error:", np.mean(np.square(errors)))

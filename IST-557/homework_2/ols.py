import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm


# load the data
df = pd.read_csv("../../hw2_data/data_train_hw2_problem2.csv")
print("shape of dataset= ", df.shape)

# find the correlation matrix and also plot it
print(df.corr())
sns.heatmap(df.corr(), cmap="YlGnBu")

# extract out X and y from the data
X = df[["x1","x2","x3"]].to_numpy()
y = df[["y"]].to_numpy()
print(X.shape, y.shape)

# calculate XTX
print(np.matmul(np.transpose(X),X))

# calculate eigen values and vectors
w,v = np.linalg.eig(np.matmul(np.transpose(X),X))
print("Eigen Values = ", w)
print("Eigen Vectors = ", v)

# fit the ols model from statsmodel library
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
model.summary()


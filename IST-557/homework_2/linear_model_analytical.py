import numpy as np
import sklearn 
import pandas as pd

#load the training data
data = pd.read_csv("../../hw2_data/data_train_hw2_problem1.csv")

X_train = data[["x1","x2","x3"]].to_numpy()
Y_train = data[["y"]].to_numpy()
print(X_train.shape, Y_train.shape)

#load the testing data
test_data = pd.read_csv("../../hw2_data/data_test_hw2_problem1.csv")
X_test = test_data[["x1","x2","x3"]].to_numpy()
Y_test = test_data[["y"]].to_numpy()
print(X_test.shape, Y_test.shape)

# calculate the analytical solution of beta
leftTerm = np.linalg.inv(np.transpose(X_train) @ X_train)
beta = leftTerm * np.transpose(X_train)
print(beta)

# predict y for training data
Y_train_pred = X_train @ beta
print(Y_train_pred.shape)

# calculate the analytical solution of variance
variance = (4.0/X_train.shape[0])*(np.sum((Y_train-Y_train_pred)**2))
print(variance)

mse_loss = sklearn.metrics.mean_squared_error(Y_train, Y_train_pred)
print(mse_loss)

Y_test_pred = X_test @ beta
print(Y_test_pred.shape)

mse_loss_test = sklearn.metrics.mean_squared_error(Y_test, Y_test_pred)
print(mse_loss_test)
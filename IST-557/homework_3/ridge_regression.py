import numpy as np 
import seaborn as sns 
import pandas as pd 
from sklearn.model_selection import KFold 
from sklearn.linear_model import Ridge, LinearRegression 
from sklearn.metrics import mean_squared_error 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler 

train_data = pd.read_csv("../../hw3_data/data_train_hw3_problem1.csv") 
print(train_data.describe()) 
X = train_data.iloc[:,1:] 
Y = train_data.iloc[:,0] 
 
X.shape, Y.shape 
kf = KFold(n_splits=10) 
 
alpha_values = [10**-30, 10**-20, 10**-10, 10**-5, 10**-2, 10**-1, 1, 5, 10 , 15, 25, 25, 75, 100] 
errors = [] 
 
for train_index, test_index in kf.split(X, Y): 
    X_train, X_val = X.iloc[train_index,:], X.iloc[test_index,:] 
    Y_train, Y_val = Y.iloc[train_index], Y.iloc[test_index] 
    fold_error = [] 
    for alpha_value in alpha_values: 
        model_R = Ridge(alpha=alpha_value) 
        model_LR = LinearRegression() 
 
        model_R.fit(X_train, Y_train) 
        model_LR.fit(X_train, Y_train) 
 
        val_error = mean_squared_error(Y_val, model_R.predict(X_val)) 
        train_val_error = mean_squared_error(Y_train, model_R.predict(X_train)) 
 
        alpha_error = [train_val_error, val_error] 
        fold_error.append(alpha_error) 
    errors.append(fold_error) 
 
errors = np.array(errors) 
mean_errors_train_val = [] 
for alpha_value in range(len(alpha_values)): 
    sum_accross_alphas = 0.0 
    for fold in range(10): 
        sum_accross_alphas += errors[fold][alpha_value][0] 
    mean_errors_train_val.append(sum_accross_alphas/10.0) 
 
mean_errors_val = [] 
for alpha_value in range(len(alpha_values)): 
    sum_accross_alphas = 0.0 
    for fold in range(10): 
        sum_accross_alphas += errors[fold][alpha_value][1] 
    mean_errors_val.append(sum_accross_alphas/10.0) 
 
default_x_ticks = range(len(alpha_values)) 
plt.plot(default_x_ticks, mean_errors_train_val) 
plt.xticks(default_x_ticks, alpha_values) 
plt.ylim(0,40) 
plt.ylabel("Mean Train MSE") 
plt.xlabel("Alpha Values") 
plt.show() 

default_x_ticks = range(len(alpha_values)) 
plt.plot(default_x_ticks, mean_errors_val) 
plt.xticks(default_x_ticks, alpha_values) 
plt.ylabel("Mean Test MSE") 
plt.xlabel("Alpha Values") 
plt.ylim(0,40) 
plt.show() 
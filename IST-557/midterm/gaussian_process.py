import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ExpSineSquared 
from sklearn.gaussian_process import GaussianProcessRegressor 
 
data = pd.read_csv("../../midterm_data/data_train_midterm_problem1.csv") 
 
data["date"] = pd.to_datetime(data["date"],format='%Y-%m-%d') 
data = data[["date", "stock_price"]].set_index("date") 
 
test_data = pd.read_csv("../../midterm_data/data_test_midterm_problem1.csv") 
test_data["date"] = pd.to_datetime(test_data["date"],format='%Y-%m-%d') 
test_data = test_data[["date"]].set_index("date") 

X = (data.index.year + data.index.month/12 + data.index.day/365).to_numpy().reshape(-1,1) 
y = np.log(data["stock_price"].to_numpy()) 
X_test = (test_data.index.year + test_data.index.month/12 + test_data.index.day/365).to_numpy().reshape(-1,1) 
 
longrange_kernel = 10 * RBF(length_scale=2) 
periodicity_kernel = 5 * RBF(length_scale=10) * ExpSineSquared(length_scale=0.05) 
noise_kernel = RBF(length_scale=2) 
 
kernel = (longrange_kernel + noise_kernel + periodicity_kernel) 
gaussian_process = GaussianProcessRegressor(kernel=kernel, optimizer=None, normalize_y=False, random_state=42) 
gaussian_process.fit(X, y - y.mean()) 
 
mean_prediction_train, std_prediction_train = gaussian_process.predict(X.reshape(-1,1), return_std=True) 
mean_prediction_test, std_prediction_test = gaussian_process.predict(X_test.reshape(-1,1), return_std = True) 
plt.scatter(X, y, label="training data", alpha=0.4) 
plt.plot(X, mean_prediction_train + y.mean(), label="Mean prediction train",color='black', alpha=0.8) 
plt.plot(X_test, mean_prediction_test + y.mean(), label="Mean prediction test",color='red', alpha=0.8) 
plt.fill_between( 
    X_test.ravel(), 
    mean_prediction_test + y.mean() - 1.96 * std_prediction_test, 
    mean_prediction_test + y.mean() + 1.96 * std_prediction_test,
    alpha=0.5, 
    label=r"95% confidence interval", 
    color="orange" 
) 
plt.legend() 
results = [] 
for i in range(mean_prediction_test.shape[0]): 
    results.append(np.random.uniform(np.exp(mean_prediction_test[i] + y.mean() - 1.96 * std_prediction_test[i]), 
                                     np.exp(mean_prediction_test[i] + y.mean() + 1.96 * std_prediction_test[i]), 
                                     1000).tolist()) 
results = np.array(results) 
results = pd.DataFrame(data=results) 
results.insert(0,"mean_preds", np.exp(mean_prediction_test + y.mean())) 
results.to_csv("./saxena_forecast_midterm_problem1.csv", header=False, index=False)
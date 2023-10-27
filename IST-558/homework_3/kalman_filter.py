import pandas as pd
import matplotlib.pyplot as plt
import pykalman
import numpy as np

data = pd.read_csv("./hw3p2_data.csv")
data.head()

n_timesteps = 200
transition_matrix = 0.98
observation_matrix = np.reshape(data.x.values,(n_timesteps,1,1))
transition_covariance = 0.15
observation_covariance = 3
initial_state_mean = 0
initial_state_covariance = 1
random_state = 1234



kf = pykalman.KalmanFilter(transition_matrices = transition_matrix, 
                           observation_matrices =  observation_matrix, 
                           transition_covariance = transition_covariance, 
                           observation_covariance = observation_covariance, 
                           initial_state_mean = initial_state_mean, 
                           initial_state_covariance = initial_state_covariance, 
                           random_state = random_state)

measurements = data.y.values
print(measurements.shape)

(filtered_state_means, filtered_state_covariances) = kf.filter(measurements)
(smoothed_state_means, smoothed_state_covariances) = kf.smooth(measurements)

plt.plot(np.arange(0,200,1), filtered_state_means, 'blue')
plt.plot(np.arange(0,200,1), smoothed_state_means, 'orange')

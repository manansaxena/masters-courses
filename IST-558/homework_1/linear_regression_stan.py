import arviz as az
import pandas as pd
import stan 

# the model defined in stan language
linear_regression = """
data {
    int<lower=0> N;
    matrix[N,3] x;
    vector[N] y;
    vector[5] mu;
    int tau;
}

parameters {
    vector[3] beta;
}

model {
    y ~ normal(x*beta,sigma);
    beta ~ normal(mu, tau);
}
"""

data = pd.read_csv("./data/hw1_data_p1.csv")

x = data.iloc[:,1:4].values
y = data.iloc[:,0].values

input_data = {"N": 10,
              "y": y,
              "x": x,
              "mu":[3,3,3],
              "tau":1}

posterior = stan.build(linear_regression, data=input_data, random_seed=1)

fit = posterior.sample(num_chains=4, num_samples=9000)
print(az.summary(fit))

inference_data = az.from_pystan(fit=fit)
az.plot_trace(inference_data)

dataFrame = fit.to_frame()

dataFrame.to_csv("./stan_problem_1.csv")

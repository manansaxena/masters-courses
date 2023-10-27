import arviz as az
import pandas as pd
import stan 
from matplotlib import pyplot as plt

mixed_effects = """
data {
    int<lower=0> N;
    int<lower=0> K;

    matrix[K,2] x;
    array[K] int y;
    array[N] int z;

    vector[2] mu;
    int b;
}

parameters {
    vector[N] phi;
    matrix[N,2] beta;
    real<lower=0> sigma;
}

model {
    for(i in 1:N){
        y[i] ~ poisson(exp(phi[i]));
        phi[i] ~ normal(x[i] * to_vector(beta[z[i]]));
    }
    for (i in 1:K) {
        beta[i] ~ normal(mu, tau);
    }
    sigma ~ inv_gamma(a,b); 
}
"""
data = pd.read_csv("./data/hw1_data_p2.csv")

x = data.iloc[:,2:4].values
y = data.iloc[:,0].values
z = data.iloc[:,1].values


input_data = {"N": x.shape[0],
              "K": max(z),
              "y": y,
              "x": x,
              "z": z,
              "mu":[3,3],
              "tau":1,
              "a":1,
              "b":1}

posterior = stan.build(mixed_effects, data=input_data, random_seed=1)

fit = posterior.sample(num_chains=4, num_samples=20000)

print(az.summary(fit))

inference_data = az.from_pystan(posterior=fit)
inference_data.to_json("./inference_data_problem_2.json")

dataFrame = fit.to_frame()

dataFrame.to_csv("./stan_problem_2.csv")

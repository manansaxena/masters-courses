# import arviz as az
import pandas as pd
import numpy as np
from scipy.stats import invgamma
from matplotlib import pyplot as plt
import seaborn as sns

def normalPDF(x, mean, standard_deviation):
    numerator = np.exp(-1/2*((x-mean)/standard_deviation)**2)
    denominator = np.sqrt(2*np.pi)*standard_deviation
    return numerator/denominator

def invGammaPDF(x, a, b):
    non_zero = int(x>=0)
    func = x**(a-1)*np.exp(-x/b)
    return non_zero*func

# generate the proposal state
def proposal(beta_s, sigma_s, delta):
    return np.random.normal(loc=np.concatenate((beta_s, sigma_s), axis=None), scale= delta, size=4)

# calculate the log_probability of a target distribution
def log_probability(X, Y, beta, sigma, mu, tau, a, b, beta1, sigma1):
    log_likelihood = np.log(invGammaPDF(sigma, a, b))
    log_likelihood += np.log(normalPDF(beta[0], mu, tau))
    log_likelihood += np.log(normalPDF(beta[1], mu, tau))
    log_likelihood += np.log(normalPDF(beta[2], mu, tau))
    for x, y in zip(X, Y):
        log_likelihood += np.log(normalPDF(y, beta.T @ x, sigma))
    log_likelihood += np.log(normalPDF(beta1[0],0,1))
    log_likelihood += np.log(normalPDF(beta1[1],0,1))
    log_likelihood += np.log(normalPDF(beta1[2],0,1))
    log_likelihood += np.log(normalPDF(sigma1,0,1))
    return log_likelihood

def run_mh(X, Y, start_value, num_hops = 10000):

    samples = np.zeros((1,4))
    samples[0,:] = start_value

    for i in range(1,num_hops+1):
        param_proposal = proposal(samples[i-1,0:3], samples[i-1,3], delta=0.1).reshape(4,1)
        param_proposal_2 = proposal(param_proposal[0:3], param_proposal[3], delta=0.1).reshape(4,1)
        ratio = log_probability(X,Y, param_proposal[0:3], 
                                param_proposal[3], 3,
                                1, 1, param_proposal_2[3]
                ) - log_probability(X, Y, samples[i-1,0:3], 
                                3,
                                1, 1, 1, param_proposal[0:3], param_proposal[3]
                ) 
        uniform_random = np.random.uniform(0,1,1)
        if (uniform_random <= min(1,np.exp(ratio))):
            samples = np.concatenate((samples, param_proposal.reshape(1,2)), axis=0)
        else:
         samples = np.concatenate((samples, samples[i-1,:].reshape(1,5)), axis=0)
    
    return samples
    

data = pd.read_csv("./data/hw1_data_p1.csv")
x = data.iloc[:,1:4].values
y = data.iloc[:,0].values
samples = run_mh(x, y, np.array([0.1,0.1,0.1,0.1]))

fig, axs = plt.subplots(2, 2)
axs[0, 0].scatter(np.arange(0,30001,1),samples[:,0])
axs[0, 0].set_title("beta 1")
axs[0, 1].scatter(np.arange(0,30001,1),samples[:,1])
axs[0, 1].set_title("beta 2")
axs[1, 0].scatter(np.arange(0,30001,1),samples[:,2])
axs[1, 0].set_title("beta 3")
axs[1, 1].scatter(np.arange(0,30001,1),samples[:,3])
axs[1, 1].set_title("sigma")
fig.tight_layout(pad=1.0)
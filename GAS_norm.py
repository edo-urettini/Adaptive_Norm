from scipy.optimize import minimize
import numpy as np

#Score Driven Normalization Gaussian Process
#The regularization choices are Full (natural gradient descent) and Root (normalized gradient descent)

#The function takes the single time series to normalize, the training data to fit the static parameters and outputs
#the normalized time series and the lists of mean and variances
#The mode (predict or update) let us choose if we want to normalize the data with the previous prediction or with the current updated parameters

def SD_Normalization_Gaussian(y, y_train, regularization, mode='predict'):
    alpha_mu, alpha_sigma, mu_0, sigma2_0 = Optimized_params_Gaussian(y_train, regularization)

    T = len(y)
    mu_list, sigma2_list = np.zeros(T), np.ones(T)
    y_normalized = np.zeros(T)

    for t in range(0, T):
        if t == 0:
            #At the first step, we update starting from the inizialization parameters
            mu_list[t], sigma2_list[t] = Update_function_Gaussian(regularization, y[t], mu_0, sigma2_0, alpha_mu, alpha_sigma)
        else:
            mu_list[t], sigma2_list[t] = Update_function_Gaussian(regularization, y[t], mu_list[t-1], sigma2_list[t-1], alpha_mu, alpha_sigma)
        
        if mode == 'predict':
            y_normalized[t] = (y[t]-mu_list[t-1])/np.sqrt(sigma2_list[t-1])
        elif mode == 'update':
            y_normalized[t] = (y[t]-mu_list[t])/np.sqrt(sigma2_list[t])
        else:
            print('Error: mode must be predict or update')
    
    return mu_list, sigma2_list, y_normalized

#Define the Update function for the Gaussian parameters. It updates the mean and the variance at each new observation
def Update_function_Gaussian(regularization, y_t, mu_t, sigma2_t, alpha_mu, alpha_sigma):
    if regularization == 'Full':
        mu_updated= mu_t + alpha_mu*(y_t-mu_t) 
        sigma2_updated= sigma2_t + alpha_sigma*((y_t-mu_t)**2  - sigma2_t)

    elif regularization == 'Root':
        mu_updated =  alpha_mu * (y_t - mu_t) / np.sqrt(sigma2_t) + mu_t
        sigma2_updated =  alpha_sigma * (-np.sqrt(2)/2 + np.sqrt(2)*(y_t-mu_t)**2 / (2*sigma2_t)) + sigma2_t
        
    else:
        print('Error: regularization must be Full or Root')
    return mu_updated, sigma2_updated


#Define the likelihood function for the Gaussian case
def neg_log_likelihood_Gaussian(params, y, regularization):
    epsilon = 1e-8
    alpha_mu, alpha_sigma, mu_0, sigma2_0 = params

    T = len(y)
    mu_list, sigma2_list = np.zeros(T), np.zeros(T)
    log_likelihood_list = np.zeros(T)
    y = np.append(y, y[T-1])

    for t in range(0, T):
        if t == 0:
            #At the first step, we update starting from the inizialization parameters
            mu_list[t], sigma2_list[t] = Update_function_Gaussian(regularization, y[t], mu_0, sigma2_0, alpha_mu, alpha_sigma)
            
        else:
            mu_list[t], sigma2_list[t] = Update_function_Gaussian(regularization, y[t], mu_list[t-1], sigma2_list[t-1], alpha_mu, alpha_sigma)
        
        log_likelihood_list[t] = -0.5 * np.log(2 * np.pi * sigma2_list[t]) - 0.5 * (y[t+1] - mu_list[t]) ** 2 / sigma2_list[t] 
    
        
    neg_log_lokelihood = -np.sum(log_likelihood_list)

    return neg_log_lokelihood/T


#Define the optimization function that optimize the likelihood function

def Optimized_params_Gaussian(y, regularization, initial_guesses= np.array([0.001, 0.001, 0, 1])):

    bounds = ((0, 1), (0, 1), (None, None), (0.00001, 1))
    optimal = minimize(lambda params: neg_log_likelihood_Gaussian(params, y, regularization), x0=initial_guesses, bounds=bounds)
    
    alpha_mu, alpha_sigma, mu_0, sigma2_0 = optimal.x
    print('Optimal parameters:  alpha_mu = {},  alpha_sigma = {}, mu_0 = {}, sigma2_0 = {}'.format(alpha_mu, alpha_sigma, mu_0, sigma2_0))

    return alpha_mu, alpha_sigma, mu_0, sigma2_0
  
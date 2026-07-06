# Functions for sampling stock paths, correlated, coupled

import numpy as np
import pandas as pd
from scipy.linalg import cholesky
from scipy.stats import norm
from scipy.stats import t as t_dist


def multivariate_samples_correlation_sample(samples, correlations):
    """
    Function to sample from multivariate normal distribution from a correlation matrix.
    """
    correlations_chol = cholesky(correlations.values, lower = True)

    n, m = correlations.shape

    normals = np.random.standard_normal(size = (n, samples))

    return correlations_chol @ normals

def multivariate_samples_root_sample(samples, correlation_root):
    """
    Function to sample from multivariate normal distribution from a correlation matrix S
    Given matrix A, the correlation root such that AA^T = S. Typically the cholesky decomposition.
    """
    n, m = correlation_root.shape

    normals = np.random.standard_normal(size = (n, samples))

    return correlation_root @ normals

def t_log_returns(multivariate_sampler, stocks, mus, sigmas, degrees_of_freedoms):
    """
    Function to take a set of multivariate Gaussian samples, and return t-distribution samples
    distributed according to the Gaussian copula implied by the multivariate Gaussian samples.
    
    Multivariate samples are standardised multivariate samples
    Stocks are the names of the stocks sampled
    mus are the daily average drifts for the stocks
    sigmas are the daily volatilities for the stocks
    degrees of freedoms are the degrees of freedom for each stock
    """
    n = len(stocks)

    log_returns = np.zeros(shape = multivariate_sampler.shape)
    
    for i in range(0, n):
        log_returns[i, :] = t_dist.ppf(norm.cdf(multivariate_sampler[i, :]), df = degrees_of_freedoms[i]) * sigmas[i] + mus[i]

    return log_returns

def generate_t_increment_portfolio_changes_paths_frame(stocks, mus, sigmas, degrees_of_freedoms, correlations, MPOR, initial_values, samples):
    """
    Given a portfolio of stocks, daily drifts, daily volatilities, degrees of freedom, MPOR, 
    initial values for stock prices, and quantities of shares held returns a samples of 
    portfolio value changes over the MPOR distributing the stocks according to t-distributions
    correlated according to a Gaussian copula with correlation matrix correlations.

    stocks: A list containing strings for the name of each stock
    mus: The daily drifts of each stock A np array of floats
    sigmas: The daily vols of each stock. A np array of floats
    degrees_of_freedoms: The t-distribution number of degrees of freedom for each stock. A fnp array of floats
    correlations: The correlation matrix of the underlying Gaussian copula. A np array or pandas matrix
    MPOR: The margin period of risk. An integer
    initial_values: The starting stock price of each stock. A np array of floats.
    quantities: The quantity of each stock held (long or short) A np array of floats.
    samples: The number of samples to generate
    """
    
    no_of_stocks = len(stocks)

    increments = np.zeros(shape = (samples, no_of_stocks))

    for i in range(0, samples):
        gaussian_samples = multivariate_samples_correlation_sample(MPOR, correlations)

        t_log_returns_array = t_log_returns(gaussian_samples, stocks, mus, sigmas, degrees_of_freedoms)

        overall_returns = t_log_returns_array.sum(axis = 1)

        increments[i, :] = (np.exp(overall_returns) - 1) * initial_values
    
    portfolio_increment_frame = pd.DataFrame(increments, columns = stocks)

    portfolio_increment_frame['Total'] = portfolio_increment_frame.sum(axis = 1)

    return portfolio_increment_frame


def generate_correlated_brownians(samples, time_steps, correlated_root, maturity):
    """
    Genereates correlated brownians according to a correlation matrix's root
    with maturity T, and time_steps time_steps. Returns samples samples of
    these paths.

    Return is a (samples, time_steps, dimension_of_correlations_matrix) np
    array, so that [i,j,k] is the k'th brownian at the j'th timestep of the i'th
    sample
    """

    n = correlated_root.shape[0] # Dimensions of correlations matrix
    brownian_intervals = np.sqrt(maturity / time_steps) * np.random.standard_normal(size = (time_steps, samples, n))
    correlated_intervals = np.einsum('stj, ij', brownian_intervals, correlated_root).T
    print(correlated_intervals.shape)
    brownian_increments = np.concatenate([np.zeros((samples, 1, n)), correlated_intervals], axis = 1)
    return brownian_increments.cumsum(axis = 1)

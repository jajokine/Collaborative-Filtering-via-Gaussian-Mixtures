# ---------------------------------------------------------------------------------------------#
#                                                                                              #
#                          Mixture Model for Collaborative Filtering                           #
#                                                                                              #                                                                                         #                                                                                             #
# ---------------------------------------------------------------------------------------------#


from typing import NamedTuple, Tuple
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, Arc


class GaussianMixture(NamedTuple):
    """Tuple holding a gaussian mixture"""
    
    mu: np.ndarray                                  # (K, d) array - each row corresponds to a gaussian component mean
    var: np.ndarray                                 # (K, ) array - each row corresponds to the variance of a component
    p: np.ndarray                                   # (K, ) array = each row corresponds to the weight of a component


def init(X: np.ndarray, K: int, seed: int = 0) -> Tuple[GaussianMixture, np.ndarray]:
    """Initializes the mixture model with random points as initial means and uniform assingments
    
    Args:
        X: (n, d) array holding the data
        K: number of components
        seed: random seed
    
    Returns:
        mixture: the initialized gaussian mixture
        post: (n, K) array holding the soft counts for all components for all samples
    """
    
    np.random.seed(seed)                            # Setting random state to seed number
    n, _ = X.shape                                  # Number of samples
    p = np.ones(K) / K

    mu = X[np.random.choice(n, K, replace=False)]   # Select K random points as initial means
    var = np.zeros(K)                               # Initialize variance components to zeros
    
    for j in range(K):                              # For loop to compute each variance component
        var[j] = ((X - mu[j])**2).mean()

    mixture = GaussianMixture(mu, var, p)           # Mixture components
    post = np.ones((n, K)) / K                      # Initializing posterior components to ones

    return mixture, post


def plot(X: np.ndarray, mixture: GaussianMixture, post: np.ndarray, title: str):
    """Plots the mixture model for 2D data"""
    
    _, K = post.shape

    percent = post / post.sum(axis=1).reshape(-1, 1)
    
    fig, ax = plt.subplots()
    ax.title.set_text(title)
    ax.set_xlim((-20, 20))
    ax.set_ylim((-20, 20))
    r = 0.25
    color = ["r", "b", "k", "y", "m", "c", "orange", "royalblue", "lime", "deeppink", "darkred", "darkviolet"]
    
    for i, point in enumerate(X):
        theta = 0
        for j in range(K):
            offset = percent[i, j] * 360
            arc = Arc(point,
                      r,
                      r,
                      0,
                      theta,
                      theta + offset,
                      edgecolor=color[j])
            ax.add_patch(arc)
            theta += offset
    
    for j in range(K):
        mu = mixture.mu[j]
        sigma = np.sqrt(mixture.var[j])
        circle = Circle(mu, sigma, color=color[j], fill=False)
        ax.add_patch(circle)
        legend = "mu = ({:0.2f}, {:0.2f})\n stdv = {:0.2f}".format(
            mu[0], mu[1], sigma)
        ax.text(mu[0], mu[1], legend)
    
    plt.axis('equal')
    plt.show()


def rmse(X, Y):
  """ Computes the root-mean-square error (RMSE) which is calculated with the standard deviation of the residuals
  (i.e. measure of difference between samples and predicted values)
  """
  
    return np.sqrt(np.mean((X - Y)**2))


def bic(X: np.ndarray, mixture: GaussianMixture, log_likelihood: float) -> float:
    """Computes the Bayesian Information Criterion for a mixture of gaussians
    BIC: log-likelihood - (1/2) * p * log(n), where n is the number of samples and p the number of free parameters
    BIC: n * log(RSS / n) + k * log(n)
    
    Args:
        X: (n, d) array holding the data
        mixture: a mixture of spherical gaussian
        log_likelihood: the log-likelihood of the data (the maximized value of the likelihood function of the model)
    
    Returns:
        float: the BIC for this mixture
    """
    
    n = X.shape[0]                                          # Number of samples

    p = 0                                                   # Initializing free parameters
    for i in range(len(mixture)):                           # For loop to go through the parameters of the mixture components 
        if i == 0:
            p += mixture[i].shape[0] * mixture[i].shape[1]  # Means, K x d parameters
        else:
            p += mixture[i].shape[0]                        # Pi and variance, K parameters
    p = p - 1                                               # Number of free (adjustable) parameters = total parameters - 1

    bic = log_likelihood - (p * np.log(n)) / 2.0            # log-likelihood - (1/2) * p * log(n)

    return bic

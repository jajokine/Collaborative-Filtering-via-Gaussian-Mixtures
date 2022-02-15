# ---------------------------------------------------------------------------------------------#
#                                                                                              #
#                                 Mixture Model Using EM (simple)                              #
#                                                                                              #                                                                                         #                                                                                             #
# ---------------------------------------------------------------------------------------------#



from typing import Tuple
import numpy as np
from common import GaussianMixture


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component
    
    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture (mu, var, pi)
    
    Returns:
        np.ndarray: (n, K) array holding the soft counts for all components for all examples
        float: log-likelihood of the assignment
    """
    
    n, d = X.shape                                             # Number of samples, dimensions
    mu, var, pi = mixture                                      # Mixture components
    K = mu.shape[0]                                            # Number of mmixtures

    pre_exp = (2*np.pi*var)**(d/2)                             # (n, K)

    post = np.linalg.norm(X[:, None] - mu, ord=2, axis=2)**2   # Norm matrix
    post = np.exp(-post/(2*var))                               # Norm matrix / 2*var
    post = post/pre_exp                                        # Posterior (n, K)

    numerator = post*pi
    
    denominator = np.sum(numerator, axis=1).reshape(-1, 1)     # p(x;theta)

    post = numerator/denominator                               # p(j|i)

    log_lh = np.sum(np.log(denominator), axis=0).item()        # Log-likelihood

    return post, log_lh


def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood of the weighted dataset
    
    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts for all components for all examples
    
    Returns:
        GaussianMixture: the new gaussian mixture
    """
    
    n, d = X.shape                                              # Numer of samples, dimensions
    K = post.shape[1]                                           # Numer of mixtures

    nj = np.sum(post, axis=0)                                   # Sum of posteriors, (K, )

    pi = nj/n                                                   # Cluster probabilities, (K, )

    mu = (post.T @ X)/nj.reshape(-1, 1)                         # Means, (K, d)

    norms = np.linalg.norm(X[:, None] - mu, ord=2,axis=2)**2    # Norms, (K, d)

    var = np.sum(post*norms, axis=0)/(nj*d)                     # Variance, shape is (K, )

    return GaussianMixture(mu, var, pi)


def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model
    
    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples
    
    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts for all components for all examples
        float: log-likelihood of the current assignment
    """

    old_log_lh = None
    new_log_lh = None  # Keep track of log-likelihood to check convergence

    # Start the main loop
    while old_log_lh is None or (new_log_lh - old_log_lh > 1e-6*np.abs(new_log_lh)):

        old_log_lh = new_log_lh

        # E-step
        post, new_log_lh = estep(X, mixture)

        # M-step
        mixture = mstep(X, post)

    return mixture, post, new_log_lh

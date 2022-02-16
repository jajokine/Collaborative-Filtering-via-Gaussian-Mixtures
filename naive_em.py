# ---------------------------------------------------------------------------------------------#
#                                                                                              #
#                                 Mixture Model Using EM (simple)                              #
#                                                                                              #                                                                                         #                                                                                             #
# ---------------------------------------------------------------------------------------------#



from typing import Tuple
import numpy as np
from common import GaussianMixture


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component.
       Compared to the K-means, the model considers probability distributions (i.e. densities/concentrations
       of the datapoints) when making the clustering decision (soft assignment) instead of just measuring the distance to each point
       towards the cluster center (hard assignment). This ca lead to points being labeled to multiple different
       clusters at the same time instead of each datapoint being part of solely one cluster.
       
    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture (mu, var, pi)
    
    Returns:
        np.ndarray: (n, K) array holding the soft counts for all components for all examples
        float: log-likelihood of the assignment
    """
    
    n, d = X.shape                                             # Number of samples, dimensions
    mu, var, pi = mixture                                      # Mixture components
    K = mu.shape[0]                                            # Number of mixture components

    pre_exp = (2*np.pi*var)**(d/2)                             # (n, K)

    post = np.linalg.norm(X[:, None] - mu, ord=2, axis=2)**2   # Norm matrix
    post = np.exp(-post/(2*var))                               # Norm matrix / 2*var
    post = post / pre_exp                                      # Posterior (n, K)

    numerator = post * pi                                      # Posterior * mixing proportions
    
    denominator = np.sum(numerator, axis=1).reshape(-1, 1)     # Sum over all posteriors

    post = numerator/denominator                               # p(j|x^(i))

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
    K = post.shape[1]                                           # Numer of mixture components

    nj = np.sum(post, axis=0)                                   # Sum of posteriors, (K, )

    pi = nj/n                                                   # Cluster probabilities, (K, )

    mu = (post.T @ X)/nj.reshape(-1, 1)                         # Means, (K, d)

    norms = np.linalg.norm(X[:, None] - mu, ord=2,axis=2)**2    # Norms, (K, d)

    var = np.sum(post*norms, axis=0)/(nj*d)                     # Variances, (K, )

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

    old_ll = None
    new_ll = None  
    
    while old_ll is None or (new_ll - old_ll > 1e-6 * np.abs(new_ll)):

        old_ll = new_ll

        # E-step
        post, new_ll = estep(X, mixture)

        # M-step
        mixture = mstep(X, post)

    return mixture, post, new_ll

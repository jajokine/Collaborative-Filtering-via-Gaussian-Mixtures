# ---------------------------------------------------------------------------------------------#
#                                                                                              #
#                    Mixture Model Using EM  (full version for matrix completion)              #
#                                                                                              #                                                                                         #                                                                                             #
# ---------------------------------------------------------------------------------------------#



from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from common import GaussianMixture


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component
    Since most of the entries are missing, the focus of the model during training is on the observed portion (i.e. using an indicator function). 
    Also, since we are dealing with a large high-dimensional data set, most of the computations should be performed with logarithms.
    
    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0) - rows are users (u) and columns are movie ratings (i) - X[u,i]
        mixture: the current gaussian mixture
    
    Returns:
        np.ndarray: (n, K) array holding the soft counts for all components of all samples
        float: log-likelihood of the assignment
    """
    
    n, d = X.shape                                                                                                      # Numer of rows, columns
    mu, var, pi = mixture                                                                                               # Mixture components
    K = mu.shape[0]                                                                                                     # Number of mixture components (i.e. clusters)

    identity_matrix = X.astype(bool).astype(int)                                                                        # Boolean matrix of the data for updating

    f = (np.sum(X**2, axis=1)[:, None] + (identity_matrix @ mu.T**2) - 2 * (X @ mu.T)) / (2 * var)                      # Exponent term: norms / 2 * variance
    pre_exp = (-np.sum(identity_matrix, axis=1).reshape(-1, 1) / 2.0) @ (np.log((2 * np.pi * var)).reshape(-1, 1)).T    # Pre-exponent term, (n, K)
    f = pre_exp - f

    f = f + np.log(pi + 1e-16)                                                                                          # f(u,j) matrix 

    logsums = logsumexp(f, axis=1).reshape(-1, 1)                                                                       # LogSumExp for summing logs of exp components
    log_post = f - logsums                                                                                              # Log-posterior

    log_lh = np.sum(logsums, axis=0).item()                                                                             # Log-likelihood

    return np.exp(log_post), log_lh                                                                                     # Return posterior and log-likelihood


def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture, min_variance: float = .25) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood of the weighted dataset
    Since there is no regularization in this model, the means are updated only when the posterior is at least one
    to avoid erratic results (i.e. to avoid that small number of points will determine the value) and
    variances have a threshold of 0.25 to avoid shrinking to zero if too few points determine them.
    
    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts for all components for all examples
        mixture: the current gaussian mixture
        min_variance: the minimum variance for each gaussian
    
    Returns:
        GaussianMixture: the new gaussian mixture
    """
    
    n, d = X.shape                                                                                            # Number of rows, columns
    mu_hat, _, _ = mixture                                                                                    # Updated means
    K = mu_hat.shape[0]                                                                                       # Numer of mixture components

    pi_hat = np.sum(post, axis=0) / n                                                                         # Update cluster probabilities (n_hat / n)

    identity_matrix = X.astype(bool).astype(int)                                                              # Boolean matrix of the data for updating

    denominator = post.T @ identity_matrix                                                                    # (K, d)
    numerator = post.T @ X                                                                                    # (K, d)
    
    update_indices = np.where(denominator >= 1)                                                               # Check sample indices that have information 
    mu_hat[update_indices] = numerator[update_indices] / denominator[update_indices]                          # Update means, (K, d)

    denominator_var = np.sum(post * np.sum(identity_matrix, axis=1).reshape(-1, 1), axis=0)                   # Denominator for variances, (K,)

    norms = np.sum(X**2, axis=1)[:, None] + (identity_matrix @ mu_hat.T**2) - 2 * (X @ mu_hat.T)              # Norms, (K, d)

    var = np.maximum(np.sum(post * norms, axis=0) / denominator_var, min_variance)                            # Update variances with a fixed threshold of 0.25, (K,)

    return GaussianMixture(mu_hat, var_hat, pi_hat)


def run(X: np.ndarray, mixture: GaussianMixture, post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model
    
    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples
    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    
    """
    
    old_ll = None
    new_ll = None  
 
    while old_ll is None or (new_ll - old_ll > 1e-6* np.abs(new_ll)):

        old_ll = new_ll

        post, new_ll = estep(X, mixture)        # E-step

        mixture = mstep(X, post, mixture)       # M-step

    return mixture, post, new_ll


def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model
    
    Args:
        X: (n, d) array of incomplete data (incomplete entries = 0)
        mixture: a mixture of gaussians
    
    Returns
        np.ndarray: a (n, d) array with completed data
    """
    
    X_pred = X.copy()                                             # Initializing predictions
    mu, _, _ = mixture                                            # Getting means

    post, _ = estep(X, mixture)                                   # E-step to calculate posterior

    missing_indices = np.where(X == 0)                            # Check incomplete entries 
    X_pred[missing_indices] = (post @ mu)[missing_indices]        # Calculate predictions with posteriors and means

    return X_pred

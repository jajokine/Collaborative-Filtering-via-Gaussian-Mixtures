import numpy as np
import kmeans
import common
import naive_em
import em


# ---------------------------------------------------------------------------------------------#
#                                                                                              #
#                                   K-means and Naive EM                                       #
#                                                                                              #
# ---------------------------------------------------------------------------------------------#

# Load data
X = np.loadtxt("data\toy_data.txt")

# Setup lists to iterate over
K = [1, 2, 3, 4]    
seeds = [0, 1, 2, 3, 4]    

cost_Kmeans = [0, 0, 0, 0, 0]
log_likelihood_naive_EM = [0, 0, 0, 0, 0]

best_seed_Kmeans = [0, 0, 0, 0]
best_seed_naive_EM = [0, 0, 0, 0]

mixtures_Kmeans = [0, 0, 0, 0, 0]
mixtures_naive_EM = [0, 0, 0, 0, 0]

posts_Kmeans = [0, 0, 0, 0, 0]
posts_naive_EM = [0, 0, 0, 0, 0]

bic = [0., 0., 0., 0.]

for k in range(len(K)):
    for i in range(len(seeds)):
        
        # Run K-means Clustering
        mixtures_Kmeans[i], posts_Kmeans[i], cost_Kmeans[i] = kmeans.run(X, *common.init(X, K[k], seeds[i]))
        
        # Run Naive EM for Gaussian Mixtures
        mixtures_naive_EM[i], posts_naive_EM[i], log_likelihood_naive_EM[i] = naive_em.run(X, *common.init(X, K[k], seeds[i]))
    
    print("================= Clusters:", K[k], "=================")
    print("Lowest Cost (K-Means):", np.min(costs_Kmeans))
    print("Maximum Log-likelihood (EM):", np.max(log_likelihood_naive_EM))
    
    # Saving seed scores
    best_seed_Kmeans[k] = np.argmin(cost_Kmeans)
    best_seed_naive_EM[k] = np.argmax(log_likelihood_naive_EM) 
    
    # Plot K-means Clustering and EM for Gaussian Mixtures results
    common.plot(X, mixtures_Kmeans[best_seed_Kmeans[k]], posts_Kmeans[best_seed_Kmeans[k]], title="K-means Clustering")
    common.plot(X, mixtures_naive_EM[best_seed_naive_EM[k]], posts_naive_EM[best_seed_naive_EM[k]], title="EM for Gaussian Mixtures (naive)") 
    
    # BIC score
    bic[k] = common.bic(X, mixtures_naive_EM[best_seed_naive_EM[k]], np.max(log_likelihood_naive_EM))
    
print("================= BIC =================")
print("Best K for EM with Gaussian Mixtures (naive):", np.argmax(bic)+1)
print("BIC Score:", np.max(bic))


# ---------------------------------------------------------------------------------------------#
#                                                                                              #
#                           Mixture Model for Collaborative Filtering                          #
#                                                                                              #
# ---------------------------------------------------------------------------------------------#

# Load data
netflix = np.loadtxt('data\netflix_incomplete.txt')

# Setup lists to iterate over 
K = [1, 2, 3, 12]    
seeds = [0, 1, 2, 3, 4]    

log_likelihood_EM = [0, 0, 0, 0, 0]
best_seed_EM = [0, 0, 0, 0]
mixtures_EM = [0, 0, 0, 0, 0]
posts_EM = [0, 0, 0, 0, 0]
bic = [0., 0., 0., 0.]

for k in range(len(K)):
    for i in range(len(seeds)):
        
        # Run EM for Gaussian Mixtures, print and plot results      
        mixtures_EM[i], posts_EM[i], log_likelihood_EM[i] = em.run(netflix, *init(netflix, K[k], seeds[i]))
    
    print("================= Clusters:", K[k], "=================")
    print("Maximum Log-likelihood (EM):", np.max(log_likelihood_naive_EM))
    
    best_seed_EM[k] = np.argmax(log_likelihood_EM)
    bic[k] = common.bic(X, mixtures_EM[seeds[k]], np.max(log_likelihood_EM))
    print("BIC:", bic[k])
    
    plot(X, mixtures_EM[best_seed_EM[k]], posts_EM[best_seed_EM[k]], title="EM for Gaussian Mixtures")

# Make predictions to fill matrix
netflix_pred = em.fill_matrix(netflix, mixtures_EM[best_seed_EM[k]])

# Calculate RMSE by comparing predicted matrix with full matrix
netflix_gold = np.loadtxt('data\netflix_complete.txt')
print("RMSE for Predictions:" common.rmse(netflix_gold, netflix_pred)

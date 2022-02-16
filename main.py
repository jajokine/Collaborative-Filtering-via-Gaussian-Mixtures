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


X = np.loadtxt("toy_data.txt")

K = [1, 2, 3, 4]    
seeds = [0, 1, 2, 3, 4]    

cost_kMeans = [0, 0, 0, 0, 0]
log_likelihood_naive_EM = [0, 0, 0, 0, 0]

best_seed_kMeans = [0, 0, 0, 0]
best_seed_naive_EM = [0, 0, 0, 0]

mixtures_kMeans = [0, 0, 0, 0, 0]
mixtures_naive_EM = [0, 0, 0, 0, 0]

posts_kMeans = [0, 0, 0, 0, 0]
posts_naive_EM = [0, 0, 0, 0, 0]

bic = [0., 0., 0., 0.]

for k in range(len(K)):
    for i in range(len(seeds)):
        
        # Run kMeans
        mixtures_kMeans[i], posts_kMeans[i], cost_kMeans[i] = kmeans.run(X, *common.init(X, K[k], seeds[i]))
        
        # Run Naive EM
        mixtures_EM[i], posts_EM[i], costs_EM[i] = naive_em.run(X, *common.init(X, K[k], seeds[i]))
    
    print("=============== Number of Clusters:", k+1, "======================")
    print("Lowest Cost (K-Means):", np.min(costs_kMeans))
    print("Maximum Log-likelihood (EM):", np.max(log_likelihood_naive_EM))
    
    # Best seeds
    best_seed_kMeans[k] = np.argmin(cost_kMeans)
    best_seed_EM[k] = np.argmax(log_likelihood_naive_EM) 
    
    # Plot kMeans and EM results
    common.plot(X, mixtures_kMeans[best_seed_kMeans[k]], posts_kMeans[best_seed_kMeans[k]], title="kMeans")
    common.plot(X, mixtures_EM[best_seed_EM[k]], posts_EM[best_seed_EM[k]], title="EM") 
    
    # BIC score
    bic[k] = common.bic(X, mixtures_EM[best_seed_EM[k]], np.max(costs_EM))
    
print("================= BIC ====================")
print("Best K is:", np.argmax(bic)+1)
print("BIC for the best K is:", np.max(bic))


# ---------------------------------------------------------------------------------------------#
#                                                                                              #
#                           Mixture Model for Collaborative Filtering                          #
#                                                                                              #
# ---------------------------------------------------------------------------------------------#


netflix = np.loadtxt('netflix_incomplete.txt')

K = [1, 2, 3, 12]    
seeds = [0, 1, 2, 3, 4]    

log_likelihood_EM = [0, 0, 0, 0, 0]
best_seed_EM = [0, 0, 0, 0]
mixtures_EM = [0, 0, 0, 0, 0]
posts_EM = [0, 0, 0, 0, 0]
bic = [0., 0., 0., 0.]

for k in range(len(K)):
    for i in range(len(seeds)):
      
        mixtures_EM[i], posts_EM[i], costs_EM[i] = em.run(netflix, *init(netflix, K[k], seeds[i]))
    print("=============== Number of Clusters:", k+1, "======================")
    print("Maximum Log-likelihood (EM):", np.max(log_likelihood_naive_EM))
    
    best_seed_EM[k] = np.argmax(log_likelihood_EM)
    bic[k] = common.bic(X, mixtures_EM[seeds[k]], np.max(log_likelihood_EM))
    print("BIC:", bic[k])
    
    plot(X, mixtures_EM[best_seed_EM[k]], posts_EM[best_seed_EM[k]], title="EM")

# Make predictions
netflix_pred = em.fill_matrix(netflix, mixtures_EM[best_seed_EM[k]])

# Calculate RMSE
netflix_gold = np.loadtxt('netflix_complete.txt')
print("RMSE for Predictions:" common.rmse(netflix_gold, netflix_pred)

## Collaborative-Filtering-via-Gaussian-Mixtures

# MITx - MicroMasters Program on Statistics and Data Science - Machine Learning with Python

Fourth Project - Collaborative Filtering Based on Gaussian Mixture Model

The fourth project for the MIT MicroMasters Program course on Machine Learning with Python concentrated on
unsupervised learning and collaborative filtering. The task of the project was to create a Generative Model that would predict
the missing entries of a movie-based database where only a fraction of movie ratings were filled by the users (rating from 1 to 5).
To be more precise, the goal was to construct a model so that each user's rating profile would come from a Gaussian Mixture Model, in the way that
there are K types of users, and in the context of each user, we should sample a user type and then the rating profile
from the Gaussian distribution associated with the type (i.e. modeling the probability distribution of each class with a Gaussian distribution).

The mixture was estimated with the Expectation Maximization (EM) algorithm in two steps: by iterately assigning users to different types (E-step),
and subsequently re-estimating the Gaussians associated with each type (M-step). Once the right mixture were to be found,
it would be used to predict all the missing values in the movie-rating matrix.

First, a K-means model was implemented to assign each points solely to one cluster based on the means and variances of the cluster centerpoints
(i.e. hard assigning of clusters) before moving to calculating the clusters with the log-likelihoods of the datapoints (i.e. soft assigning of clusters).
The mixture was estimated with the Expectation Maximization (EM) algorithm in two steps: first, by iterately assigning users to different types (E-step),
and subsequently, re-estimating the Gaussians associated with each type (M-step).

Once the right mixture were to be found, it would be used to predict all the missing values in the movie-rating matrix. Finally, for model selection, the models were compared with the Bayesian Information Criterion (BIC) which captures the tradeoff between the log-likelihood of the data, and the number of parameters that the model uses, and to analyze the accuracy of the model, the final predicted matrix was compared to the full (given) matrix with RMSE. 

## Dataset

      - toy_data.txt : A 2D dataset (250 by 2 matrix).
      
      - netflix_incomplete.txt : Netflix dataset with missing entries to be completed (1200 by 1200 matrix with 1,111,768 missing entries).
      
      - netflix_complete.txt : Netflix dataset with missing entries completed (1200 by 1200 matrix with 1,440,000 entries).
      
      - test_incomplete.txt : Incomplete test dataset for testing code (20 by 5 matrix with 19 missing entries).
      
      - test_comlete.txt : Complete test dataset for testing code (20 by 5 matrix with 100 entries).
      
      - test_solutions.txt : Solutions for test dataset for testing code.

## Access and requirements

The file main.py runs the code with the help of the three modules that contain the different models - kmeans.py, naive_em.py and em.py, as well as the file common.py that contains the main framework that is used by all the models.

The dependencies and requirements can be seen from requirements.txt that can be installed in shell with the command:

      pip install -r requirements.txt

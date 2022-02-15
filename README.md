# Collaborative-Filtering-via-Gaussian-Mixtures
MITx - MicroMasters Program on Statistics and Data Science - Machine Learning with Python

Fourth Project - Collaborative Filtering Based on Gaussian Mixture Model

The fourth project for the MIT MicroMasters Program course on Machine Learning with Python concentrated on
unsupervised learning and collaborative filtering. The task of the project was to create a Generative Model that would predict
the missing entries of a movie-based database where only a fraction of movie ratings were filled by the users.

The goal was to construct a model so that each user's rating profile would come from a mixture model, in the way that
there are K types of users, and in the context of each user, we should sample a user type and then the rating profile
from the Gaussian distribution associated with the type (i.e. modeling the probability distribution of each class with a Gaussian distribution).
The mixture was then to be estimated with the Expectation Maximization (EM) algorithm in two steps: by iterately assigning users to types (E-step)
and subsequently re-estimating the Gaussians associated with each type (M-step). Once the right mixture were to be found,
it would be used to predict all the missing values in the movie-rating matrix.

The project started by incorporting the Generative Model first to K-means clustering and then moving towards a Gaussian Mixture Model.

Additional helper functions were given to complete the project in two weeks of time.

DATASET

      - toy_data.txt : A 2D dataset.
      
      - netflix_incomplete.txt : Netflix dataset with missing entries to be completed.
      
      - netflix_complete.txt : Netflix dataset with missing entries completed.
      
      - test_incomplete.txt : Incomplete test dataset for testing code.
      
      - test_comlete.txt : Complete test dataset for testing code.
      
      - test_solutions.txt : Solutions for test dataset for testing code.

# Recommendation.jl

**Recommendation.jl** is a Julia package for building recommender systems. Thanks to independent *data accessor* and *recommender* implementations, this package enables you to build recommendation systems on your own data and algorithms.

This package supports the following features:

- Recommendation
  - Baseline (non-personalized) recommenders based on:
    - Most frequently co-occurred items (CoOccurrence)
    - Most popular items (MostPopular)
    - Percentage of ratings which are greater than a certain threshold value (ThresholdPercentage)
    - Global user/item mean rating (UserMean, ItemMean)
  - Personalized recommenders
    - Item-based collaborative filtering (ItemKNN)
    - User-based collaborative filtering (UserKNN)
    - Singular Value Decomposition (SVD)
    - Matrix Factorization (MF)
    - Content-based recommendation using TF-IDF scoring (TFIDF)
- Evaluation
  - 5-fold cross validation
  - Rating metric
    - Root Mean Squared Error (RMSE)
    - Mean Absolute Error (MAE)
  - Ranking metric
    - Recall
    - Precision
    - Mean Average Precision (MAP)
    - Area Under the ROC curve (AUC)
    - Mean Reciprocal Rank (MRR)
    - Mean Percentile Rank (MPR)
    - Normalized Discounted Cumulative Gain (NDCG)

For more information, you can refer to [my article](http://takuti.me/note/recommendation-julia/).

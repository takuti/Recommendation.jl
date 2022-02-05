using Recommendation

data = load_movielens_100k()
recall = cross_validation(
                          1,            # N-fold
                          Recall,       # Metric
                          5,            # Top-k
                          MostPopular,  # Recommender
                          data          # Data Accessor
                         )
println(recall)

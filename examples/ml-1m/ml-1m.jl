using JLD
using Recommendation

R = load("data/ml-1m.jld")["R"]
da = DataAccessor(R)

recall = cross_validation(
                          MostPopular,  # Recommender
                          Parameters(), # Hyperparameters
                          da,           # Data Accessor
                          1,            # N-fold
                          Recall(),     # Metric
                          5             # Top-k
                         )
println(recall)

using JLD
using SparseArrays
using Recommendation

R = load("data/ml-100k.jld")["R"]
da = DataAccessor(R)

recall = cross_validation(
                          1,            # N-fold
                          Recall,       # Metric
                          5             # Top-k
                          MostPopular,  # Recommender
                          da            # Data Accessor
                         )
println(recall)

using Recommendation

# configureable parameters for cross validation
n_fold = 2
metric = Recall
topk = 5
recommender = MostPopular
data = load_movielens_100k()

res = cross_validation(n_fold, metric, topk,
                       recommender, data)
println("Average $metric = $res")

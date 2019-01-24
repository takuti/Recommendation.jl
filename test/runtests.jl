using Recommendation
using Test
using SparseArrays

include("test_base_recommender.jl")
include("test_data_accessor.jl")

include("baseline/test_user_mean.jl")
include("baseline/test_item_mean.jl")
include("baseline/test_most_popular.jl")
include("baseline/test_threshold_percentage.jl")
include("baseline/test_co_occurrence.jl")

include("model/test_tf_idf.jl")
include("model/test_user_knn.jl")
include("model/test_item_knn.jl")
include("model/test_svd.jl")
include("model/test_mf.jl")

include("metric/test_accuracy.jl")
include("metric/test_ranking.jl")

include("evaluation/test_evaluate.jl")
include("evaluation/test_cross_validation.jl")

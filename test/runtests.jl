using Recommendation
using Test
using SparseArrays
using ZipFile

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
include("model/test_matrix_factorization.jl")
include("model/test_factorization_machines.jl")

include("metrics/test_accuracy.jl")
include("metrics/test_ranking.jl")

include("evaluation/test_evaluate.jl")
include("evaluation/test_cross_validation.jl")

include("test_compat.jl")
include("test_datasets.jl")
include("test_synthetic.jl")
include("test_utils.jl")

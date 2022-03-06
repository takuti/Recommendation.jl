module Recommendation

import Statistics: mean

using LinearAlgebra
import LinearAlgebra: svd

using Random
using Downloads
using ZipFile

include("types.jl")
include("utils.jl")

include("data_accessor.jl")
include("base_recommender.jl")

include("datasets.jl")
include("synthetic.jl")

include("baseline/user_mean.jl")
include("baseline/item_mean.jl")
include("baseline/most_popular.jl")
include("baseline/threshold_percentage.jl")
include("baseline/co_occurrence.jl")

include("model/tf_idf.jl")
include("model/user_knn.jl")
include("model/item_knn.jl")
include("model/svd.jl")
include("model/matrix_factorization.jl")
include("model/factorization_machines.jl")

include("metrics/base.jl")
include("metrics/accuracy.jl")
include("metrics/ranking.jl")
include("metrics/intra_list.jl")
include("metrics/aggregated.jl")

include("evaluation/evaluate.jl")
include("evaluation/cross_validation.jl")

include("compat.jl")

end # module

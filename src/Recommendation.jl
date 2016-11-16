module Recommendation

include("types.jl")

include("base_recommender.jl")
include("data_accessor.jl")

include("baseline/user_mean.jl")
include("baseline/item_mean.jl")
include("baseline/most_popular.jl")
include("baseline/threshold_percentage.jl")
include("baseline/co_occurrence.jl")

include("model/tf_idf.jl")
include("model/user_knn.jl")
include("model/item_knn.jl")
include("model/svd.jl")
include("model/mf.jl")

include("metric/accuracy.jl")
include("metric/ranking.jl")

end # module

module Recommendation

include("recommender.jl")

include("baseline/user_mean.jl")
include("baseline/item_mean.jl")
include("baseline/most_popular.jl")
include("baseline/threshold_percentage.jl")
include("baseline/co_occurrence.jl")

include("content/tf_idf.jl")

include("rating/user_knn.jl")
include("rating/item_knn.jl")

include("utils/measures.jl")

end # module

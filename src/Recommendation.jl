module Recommendation

include("recommender.jl")

include("baseline/user_mean.jl")
include("baseline/item_mean.jl")
include("baseline/most_popular.jl")
include("baseline/threshold_percentage.jl")

include("utils/measures.jl")

end # module

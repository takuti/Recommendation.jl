using Recommendation
using Base.Test

include("test_recommender.jl")

include("baseline/test_user_mean.jl")
include("baseline/test_item_mean.jl")

include("utils/test_measures.jl")

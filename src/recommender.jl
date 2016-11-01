export Recommender
export predict, ranking

abstract Recommender

function predict(recommender::Recommender, u::Int, i::Int)
    error("prdict is not implemented for recommender type $(typeof(recommender))")
end

# Return a ranking score of item i for user u
function ranking(recommender::Recommender, u::Int, i::Int)
    error("ranking is not implemented for recommender type $(typeof(recommender))")
end

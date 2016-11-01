export Recommender
export predict

abstract Recommender

function predict(recommender::Recommender, u::Int, i::Int)
    error("prdict is not implemented for recommender type $(typeof(recommender))")
end

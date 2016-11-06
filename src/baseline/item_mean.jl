export ItemMean

immutable ItemMean <: Recommender
    m::AbstractMatrix
end

function predict(recommender::ItemMean, u::Int, i::Int)
    v = recommender.m[:, i]
    sum(v) / countnz(v)
end

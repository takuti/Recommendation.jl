export UserMean

immutable UserMean <: Recommender
    m::AbstractMatrix
end

function predict(recommender::UserMean, u::Int, i::Int)
    v = recommender.m[u, :]
    sum(v) / countnz(v)
end

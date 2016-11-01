export UserMean

immutable UserMean <: Recommender
    m::SparseMatrixCSC
end

function predict(recommender::UserMean, u::Int, i::Int)
    mean(recommender.m[u, :])
end

export UserMean

immutable UserMean <: Recommender
    m::SparseMatrixCSC
end

function predict(recommender::UserMean, u::Int, i::Int)
    v = recommender.m[u, :]
    sum(v) / nnz(v)
end

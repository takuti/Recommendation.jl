export ItemMean

immutable ItemMean <: Recommender
    m::SparseMatrixCSC
end

function predict(recommender::ItemMean, u::Int, i::Int)
    v = recommender.m[:, i]
    sum(v) / nnz(v)
end

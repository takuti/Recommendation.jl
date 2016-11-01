export ItemMean

immutable ItemMean <: Recommender
    m::SparseMatrixCSC
end

function predict(recommender::ItemMean, u::Int, i::Int)
    mean(recommender.m[:, i])
end

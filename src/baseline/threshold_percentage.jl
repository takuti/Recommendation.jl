export ThresholdPercentage

immutable ThresholdPercentage <: Recommender
    m::SparseMatrixCSC
    th::Float64
end

function ranking(recommender::ThresholdPercentage, u::Int, i::Int)
    v = recommender.m[:, i]
    v[v .>= recommender.th].n / nnz(v) * 100.0
end

export MostPopular

immutable MostPopular <: Recommender
    m::SparseMatrixCSC
end

function ranking(recommender::MostPopular, u::Int, i::Int)
    nnz(recommender.m[:, i])
end

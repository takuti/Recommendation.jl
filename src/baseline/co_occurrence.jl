export CoOccurrence

immutable CoOccurrence <: Recommender
    m::SparseMatrixCSC
    i::Int
end

function ranking(recommender::CoOccurrence, u::Int, i::Int)
    v_ref = recommender.m[:, recommender.i]
    v = recommender.m[:, i]
    v_ref[(v_ref .> 0) & (v .> 0)].n / nnz(v_ref) * 100.0
end

export CoOccurrence

immutable CoOccurrence <: Recommender
    m::SparseMatrixCSC
    v_ref::SparseVector
end

CoOccurrence(m::SparseMatrixCSC, i::Int) = begin
    CoOccurrence(m, m[:, i])
end

function ranking(recommender::CoOccurrence, u::Int, i::Int)
    v = recommender.m[:, i]
    recommender.v_ref[(recommender.v_ref .> 0) & (v .> 0)].n / nnz(recommender.v_ref) * 100.0
end

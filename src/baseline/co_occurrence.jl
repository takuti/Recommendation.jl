export CoOccurrence

immutable CoOccurrence <: Recommender
    m::AbstractMatrix
    v_ref::AbstractVector
end

CoOccurrence(m::AbstractMatrix, i::Int) = begin
    CoOccurrence(m, m[:, i])
end

function ranking(recommender::CoOccurrence, u::Int, i::Int)
    v = recommender.m[:, i]
    length(recommender.v_ref[(recommender.v_ref .> 0) & (v .> 0)]) / countnz(recommender.v_ref) * 100.0
end

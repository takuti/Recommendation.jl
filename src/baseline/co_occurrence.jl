export CoOccurrence

immutable CoOccurrence <: Recommender
    m::AbstractMatrix
    i_ref::Int
    scores::AbstractVector
end

CoOccurrence(m::AbstractMatrix, i_ref::Int) = begin
    n_user, n_item = size(m)

    v_ref = m[:, i_ref]
    c = countnz(v_ref)

    scores = zeros(n_item)

    for i in 1:n_item
        v = m[:, i]
        cc = length(v_ref[(v_ref .> 0) & (v .> 0)])
        scores[i] = cc / c * 100.0
    end

    CoOccurrence(m, i_ref, scores)
end

function ranking(recommender::CoOccurrence, u::Int, i::Int)
    recommender.scores[i]
end

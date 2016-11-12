export CoOccurrence

immutable CoOccurrence <: Recommender
    da::DataAccessor
    i_ref::Int
    scores::AbstractVector
end

CoOccurrence(da::DataAccessor, i_ref::Int) = begin
    n_user, n_item = size(da.R)

    v_ref = da.R[:, i_ref]
    c = countnz(v_ref)

    scores = zeros(n_item)

    for i in 1:n_item
        v = da.R[:, i]
        cc = length(v_ref[(v_ref .> 0) & (v .> 0)])
        scores[i] = cc / c * 100.0
    end

    CoOccurrence(da, i_ref, scores)
end

function ranking(recommender::CoOccurrence, u::Int, i::Int)
    recommender.scores[i]
end

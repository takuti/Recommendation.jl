export CoOccurrence

immutable CoOccurrence <: Recommender
    da::DataAccessor
    i_ref::Int
    scores::AbstractVector
end

CoOccurrence(da::DataAccessor, i_ref::Int) = begin
    n_item = size(da.R, 2)
    CoOccurrence(da, i_ref, zeros(n_item))
end

function build(recommender::CoOccurrence)
    n_item = size(recommender.da.R, 2)

    v_ref = recommender.da.R[:, recommender.i_ref]
    c = countnz(v_ref)

    for i in 1:n_item
        v = recommender.da.R[:, i]
        cc = length(v_ref[(v_ref .> 0) & (v .> 0)])
        recommender.scores[i] = cc / c * 100.0
    end
end

function ranking(recommender::CoOccurrence, u::Int, i::Int)
    recommender.scores[i]
end

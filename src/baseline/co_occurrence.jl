export CoOccurrence

"""

    CoOccurrence(
        da::DataAccessor,
        i_ref::Int
    )

Recommend items which are most frequently co-occurred with a reference item `i_ref`.
"""
struct CoOccurrence <: Recommender
    da::DataAccessor
    i_ref::Int
    scores::AbstractVector
    states::States

    function CoOccurrence(da::DataAccessor, i_ref::Int)
        n_item = size(da.R, 2)
        new(da, i_ref, zeros(n_item), States(:built => false))
    end
end

function build!(recommender::CoOccurrence)
    n_item = size(recommender.da.R, 2)

    v_ref = recommender.da.R[:, recommender.i_ref]
    c = count(!iszero, v_ref)

    for i in 1:n_item
        v = recommender.da.R[:, i]
        cc = length(v_ref[(v_ref .> 0) .& (v .> 0)])
        recommender.scores[i] = cc / c * 100.0
    end

    recommender.states[:built] = true
end

function ranking(recommender::CoOccurrence, u::Int, i::Int)
    check_build_status(recommender)
    recommender.scores[i]
end

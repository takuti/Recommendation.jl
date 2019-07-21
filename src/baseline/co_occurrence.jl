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

function build(rec::CoOccurrence)
    n_item = size(rec.da.R, 2)

    v_ref = rec.da.R[:, rec.i_ref]
    c = count(!iszero, v_ref)

    for i in 1:n_item
        v = rec.da.R[:, i]
        cc = length(v_ref[(v_ref .> 0) .& (v .> 0)])
        rec.scores[i] = cc / c * 100.0
    end

    rec.states[:built] = true
end

function ranking(rec::CoOccurrence, u::Int, i::Int)
    check_build_status(rec)
    rec.scores[i]
end

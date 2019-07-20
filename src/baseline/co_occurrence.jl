export CoOccurrence

"""

    CoOccurrence(
        da::DataAccessor,
        hyperparams::Parameters=Parameters(:i_ref => 1)
    )

Recommend items which are most frequently co-occurred with a reference item `i_ref`.
"""
struct CoOccurrence <: Recommender
    da::DataAccessor
    hyperparams::Parameters
    scores::AbstractVector
    states::States

    function CoOccurrence(da::DataAccessor, hyperparams::Parameters=Parameters(:i_ref => 1))
        n_item = size(da.R, 2)
        new(da, hyperparams, zeros(n_item), States(:is_built => false))
    end
end

function build(rec::CoOccurrence)
    n_item = size(rec.da.R, 2)

    v_ref = rec.da.R[:, rec.hyperparams[:i_ref]]
    c = count(!iszero, v_ref)

    for i in 1:n_item
        v = rec.da.R[:, i]
        cc = length(v_ref[(v_ref .> 0) .& (v .> 0)])
        rec.scores[i] = cc / c * 100.0
    end

    rec.states[:is_built] = true
end

function ranking(rec::CoOccurrence, u::Int, i::Int)
    check_build_status(rec)
    rec.scores[i]
end

export CoOccurrence

"""

    CoOccurrence(
        data::DataAccessor,
        i_ref::Integer
    )

Recommend items which are most frequently co-occurred with a reference item `i_ref`.
"""
struct CoOccurrence <: Recommender
    data::DataAccessor
    i_ref::Integer
    scores::AbstractVector

    function CoOccurrence(data::DataAccessor, i_ref::Integer)
        n_item = size(data.R, 2)
        new(data, i_ref, vector(n_item))
    end
end

isdefined(recommender::CoOccurrence) = isfilled(recommender.scores)

function fit!(recommender::CoOccurrence)
    v_ref = recommender.data.R[:, recommender.i_ref]
    c = sum(!iszero, v_ref)

    # count elements that are known and non-zero both in v & v_ref
    CC = vec(sum(!iszero, recommender.data.R .* v_ref, dims=1))

    recommender.scores[:] = CC / c * 100.0
end

function ranking(recommender::CoOccurrence, u::Integer, i::Integer)
    validate(recommender)
    recommender.scores[i]
end

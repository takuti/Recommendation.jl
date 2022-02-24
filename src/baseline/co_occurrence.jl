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
    # bit vector representing whether the reference item `i_ref` is rated by a user or not
    v_ref = (!iszero).(recommender.data.R[:, recommender.i_ref])

    # total num of ratings for the reference item
    c = sum(v_ref)

    # for each item `i`, count num of users who rated both `i` and `i_ref`
    CC = vec(v_ref' * (!iszero).(recommender.data.R))

    recommender.scores[:] = CC / c * 100.0
end

function ranking(recommender::CoOccurrence, u::Integer, i::Integer)
    validate(recommender)
    recommender.scores[i]
end

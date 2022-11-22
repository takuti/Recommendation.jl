export CoOccurrence

"""

    CoOccurrence(
        data::DataAccessor,
        item_ref::Integer
    )

Recommend items which are most frequently co-occurred with a reference item `item_ref`.
"""
struct CoOccurrence <: Recommender
    data::DataAccessor
    item_ref::Integer
    scores::AbstractVector

    function CoOccurrence(data::DataAccessor, item_ref::Integer)
        n_items = size(data.R, 2)
        new(data, item_ref, vector(n_items))
    end
end

isdefined(recommender::CoOccurrence) = isfilled(recommender.scores)

function fit!(recommender::CoOccurrence)
    # bit vector representing whether the reference item `item_ref` is rated by a user or not
    vec_ref = (!iszero).(recommender.data.R[:, recommender.item_ref])

    # total num of ratings for the reference item
    c = sum(vec_ref)

    # for each item `i`, count num of users who rated both `i` and `item_ref`
    CC = vec(vec_ref' * (!iszero).(recommender.data.R))

    recommender.scores[:] = CC / c * 100.0
end

function predict(recommender::CoOccurrence, user::Integer, item::Integer)
    validate(recommender)
    recommender.scores[item]
end

function predict(recommender::CoOccurrence, indices::AbstractVector{T}) where {T<:CartesianIndex{2}}
    validate(recommender)
    items = map(idx -> idx[2], indices)
    recommender.scores[items]
end

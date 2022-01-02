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

isbuilt(recommender::CoOccurrence) = isfilled(recommender.scores)

function build!(recommender::CoOccurrence)
    n_item = size(recommender.data.R, 2)

    v_ref = recommender.data.R[:, recommender.i_ref]
    c = count(!isalmostzero, v_ref)

    for i in 1:n_item
        v = recommender.data.R[:, i]
        cc = length(v_ref[(v_ref .> almost_zero) .& (v .> almost_zero)])
        recommender.scores[i] = cc / c * 100.0
    end
end

function ranking(recommender::CoOccurrence, u::Integer, i::Integer)
    check_build_status(recommender)
    recommender.scores[i]
end

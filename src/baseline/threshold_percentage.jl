export ThresholdPercentage

"""

    ThresholdPercentage(
        data::DataAccessor,
        th::AbstractFloat
    )

Recommend based on percentage of ratings which are greater than a certain threshold value `th`.
"""
struct ThresholdPercentage <: Recommender
    data::DataAccessor
    th::AbstractFloat
    scores::AbstractVector

    function ThresholdPercentage(data::DataAccessor, th::AbstractFloat)
        n_items = size(data.R, 2)
        new(data, th, vector(n_items))
    end
end

isdefined(recommender::ThresholdPercentage) = isfilled(recommender.scores)

function fit!(recommender::ThresholdPercentage)
    users_rated_higher = sum(r->r>=recommender.th, recommender.data.R, dims=1)
    users_rated = sum(!iszero, recommender.data.R, dims=1)
    recommender.scores[:] = vec(users_rated_higher ./ users_rated * 100.0)
end

function predict(recommender::ThresholdPercentage, user::Integer, item::Integer)
    validate(recommender)
    recommender.scores[item]
end

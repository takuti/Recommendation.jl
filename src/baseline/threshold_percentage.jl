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
        n_item = size(data.R, 2)
        new(data, th, vector(n_item))
    end
end

isdefined(recommender::ThresholdPercentage) = isfilled(recommender.scores)

function fit!(recommender::ThresholdPercentage)
    n_item = size(recommender.data.R, 2)

    for i in 1:n_item
        v = recommender.data.R[:, i]
        recommender.scores[i] = count(r -> r >= recommender.th, v) / count(!iszero, v) * 100.0
    end
end

function ranking(recommender::ThresholdPercentage, u::Integer, i::Integer)
    validate(recommender)
    recommender.scores[i]
end

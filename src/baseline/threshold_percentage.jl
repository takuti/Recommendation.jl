export ThresholdPercentage

"""

    ThresholdPercentage(
        data::DataAccessor,
        th::Float64
    )

Recommend based on percentage of ratings which are greater than a certain threshold value `th`.
"""
struct ThresholdPercentage <: Recommender
    data::DataAccessor
    th::Float64
    scores::AbstractVector

    function ThresholdPercentage(data::DataAccessor, th::Float64)
        n_item = size(data.R, 2)
        new(data, th, vector(n_item))
    end
end

isbuilt(recommender::ThresholdPercentage) = isfilled(recommender.scores)

function build!(recommender::ThresholdPercentage)
    n_item = size(recommender.data.R, 2)

    for i in 1:n_item
        v = recommender.data.R[:, i]
        recommender.scores[i] = length(v[v .>= recommender.th]) / count(!iszero, v) * 100.0
    end
end

function ranking(recommender::ThresholdPercentage, u::Int, i::Int)
    check_build_status(recommender)
    recommender.scores[i]
end

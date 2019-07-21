export ThresholdPercentage

"""

    ThresholdPercentage(
        da::DataAccessor,
        th::Float64
    )

Recommend based on percentage of ratings which are greater than a certain threshold value `th`.
"""
struct ThresholdPercentage <: Recommender
    da::DataAccessor
    th::Float64
    scores::AbstractVector
    states::States

    function ThresholdPercentage(da::DataAccessor, th::Float64)
        n_item = size(da.R, 2)
        new(da, th, zeros(n_item), States(:built => false))
    end
end

function build!(recommender::ThresholdPercentage)
    n_item = size(recommender.da.R, 2)

    for i in 1:n_item
        v = recommender.da.R[:, i]
        recommender.scores[i] = length(v[v .>= recommender.th]) / count(!iszero, v) * 100.0
    end

    recommender.states[:built] = true
end

function ranking(recommender::ThresholdPercentage, u::Int, i::Int)
    check_build_status(recommender)
    recommender.scores[i]
end

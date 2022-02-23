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
    M = sum(r->r>=recommender.th, recommender.data.R, dims=1)
    N = sum(!iszero, recommender.data.R, dims=1)
    recommender.scores[:] = vec(M ./ N * 100.0)
end

function ranking(recommender::ThresholdPercentage, u::Integer, i::Integer)
    validate(recommender)
    recommender.scores[i]
end

export ThresholdPercentage

immutable ThresholdPercentage <: Recommender
    da::DataAccessor
    th::Number
    scores::AbstractVector
end

ThresholdPercentage(da::DataAccessor, th::Number) = begin
    n_item = size(da.R, 2)
    ThresholdPercentage(da, th, zeros(n_item))
end

function build(recommender::ThresholdPercentage)
    n_item = size(recommender.da.R, 2)

    for i in 1:n_item
        v = recommender.da.R[:, i]
        recommender.scores[i] = length(v[v .>= recommender.th]) / countnz(v) * 100.0
    end
end

function ranking(recommender::ThresholdPercentage, u::Int, i::Int)
    recommender.scores[i]
end

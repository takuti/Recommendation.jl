export ThresholdPercentage

immutable ThresholdPercentage <: Recommender
    da::DataAccessor
    th::Number
    scores::AbstractVector
end

ThresholdPercentage(da::DataAccessor, th::Number) = begin
    n_user, n_item = size(da.R)

    scores = zeros(n_item)

    for i in 1:n_item
        v = da.R[:, i]
        scores[i] = length(v[v .>= th]) / countnz(v) * 100.0
    end

    ThresholdPercentage(da, th, scores)
end

function ranking(recommender::ThresholdPercentage, u::Int, i::Int)
    recommender.scores[i]
end

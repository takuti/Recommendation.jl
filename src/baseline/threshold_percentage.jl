export ThresholdPercentage

immutable ThresholdPercentage <: Recommender
    m::AbstractMatrix
    th::Number
    scores::AbstractVector
end

ThresholdPercentage(m::AbstractMatrix, th::Number) = begin
    n_user, n_item = size(m)

    scores = zeros(n_item)

    for i in 1:n_item
        v = m[:, i]
        scores[i] = length(v[v .>= th]) / countnz(v) * 100.0
    end

    ThresholdPercentage(m, th, scores)
end

function ranking(recommender::ThresholdPercentage, u::Int, i::Int)
    recommender.scores[i]
end

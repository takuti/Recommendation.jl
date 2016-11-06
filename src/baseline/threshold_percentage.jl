export ThresholdPercentage

immutable ThresholdPercentage <: Recommender
    m::AbstractMatrix
    th::Float64
end

function ranking(recommender::ThresholdPercentage, u::Int, i::Int)
    v = recommender.m[:, i]
    length(v[v .>= recommender.th]) / countnz(v) * 100.0
end

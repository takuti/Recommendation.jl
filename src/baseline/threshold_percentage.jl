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

function build(rec::ThresholdPercentage)
    n_item = size(rec.da.R, 2)

    for i in 1:n_item
        v = rec.da.R[:, i]
        rec.scores[i] = length(v[v .>= rec.th]) / countnz(v) * 100.0
    end
end

function ranking(rec::ThresholdPercentage, u::Int, i::Int)
    rec.scores[i]
end

export ThresholdPercentage

immutable ThresholdPercentage <: Recommender
    da::DataAccessor
    th::Number
    scores::AbstractVector
    states::States
end

ThresholdPercentage(da::DataAccessor, th::Number) = begin
    n_item = size(da.R, 2)
    ThresholdPercentage(da, th, zeros(n_item), States(:is_built => false))
end

function build(rec::ThresholdPercentage)
    n_item = size(rec.da.R, 2)

    for i in 1:n_item
        v = rec.da.R[:, i]
        rec.scores[i] = length(v[v .>= rec.th]) / countnz(v) * 100.0
    end

    rec.states[:is_built] = true
end

function ranking(rec::ThresholdPercentage, u::Int, i::Int)
    check_build_status(rec)
    rec.scores[i]
end

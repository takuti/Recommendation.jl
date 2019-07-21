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

function build(rec::ThresholdPercentage)
    n_item = size(rec.da.R, 2)

    for i in 1:n_item
        v = rec.da.R[:, i]
        rec.scores[i] = length(v[v .>= rec.th]) / count(!iszero, v) * 100.0
    end

    rec.states[:built] = true
end

function ranking(rec::ThresholdPercentage, u::Int, i::Int)
    check_build_status(rec)
    rec.scores[i]
end

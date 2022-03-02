export Metric, AccuracyMetric, RankingMetric
export measure, count_intersect, coverage

abstract type Metric end

abstract type AccuracyMetric <: Metric end

function measure(metric::AccuracyMetric, truth::AbstractVector, pred::AbstractVector)
    error("measure is not implemented for metric type $(typeof(metric))")
end

abstract type RankingMetric <: Metric end

function measure(metric::RankingMetric, truth::AbstractVector{T}, pred::AbstractVector{T}, k::Integer) where T
    error("measure is not implemented for metric type $(typeof(metric))")
end

function count_intersect(s1::Union{AbstractSet, AbstractVector}, s2::Union{AbstractSet, AbstractVector})
    length(intersect(s1, s2))
end

function coverage(items::Union{AbstractSet, AbstractVector}, catalog::Union{AbstractSet, AbstractVector})
    count_intersect(items, catalog) / length(catalog)
end

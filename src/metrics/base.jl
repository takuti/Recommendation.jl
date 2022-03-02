export Metric, AccuracyMetric, RankingMetric
export measure, count_true_positive, coverage

abstract type Metric end

abstract type AccuracyMetric <: Metric end

function measure(metric::AccuracyMetric, truth::AbstractVector, pred::AbstractVector)
    error("measure is not implemented for metric type $(typeof(metric))")
end

abstract type RankingMetric <: Metric end

function measure(metric::RankingMetric, truth::AbstractVector{T}, pred::AbstractVector{T}, k::Integer) where T
    error("measure is not implemented for metric type $(typeof(metric))")
end

function count_true_positive(truth::AbstractVector{T}, pred::AbstractVector{T}) where T
    if length(truth) == 0 || length(pred) == 0
        0
    else
        sum(in(truth), pred)
    end
end

function coverage(items::Union{AbstractSet, AbstractVector}, catalog::Union{AbstractSet, AbstractVector})
    length(intersect(items, catalog)) / length(catalog)
end

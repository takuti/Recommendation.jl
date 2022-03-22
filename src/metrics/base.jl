export Metric, AccuracyMetric, RankingMetric, IntraListMetric, AggregatedMetric
export measure

abstract type Metric end

abstract type AccuracyMetric <: Metric end
function measure(metric::AccuracyMetric, truth::AbstractVector, pred::AbstractVector)
    error("measure is not implemented for metric type $(typeof(metric))")
end

abstract type RankingMetric <: Metric end
function measure(metric::RankingMetric, truth::AbstractVector{T}, pred::AbstractVector{T}, topk::Integer) where T
    error("measure is not implemented for metric type $(typeof(metric))")
end

abstract type IntraListMetric <: Metric end
function measure(metric::IntraListMetric, recommendations::Union{AbstractSet, AbstractVector}; kwargs...)
    error("measure is not implemented for metric type $(typeof(metric))")
end

abstract type AggregatedMetric <: Metric end
function measure(metric::AggregatedMetric, recommendations::AbstractVector{<:AbstractVector{<:Integer}}; kwargs...)
    error("measure is not implemented for metric type $(typeof(metric))")
end

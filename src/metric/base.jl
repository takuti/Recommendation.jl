export Metric, AccuracyMetric, RankingMetric
export measure, count_true_positive

abstract type Metric end

abstract type AccuracyMetric <: Metric end

function measure(metric::AccuracyMetric, truth::AbstractVector, pred::AbstractVector)
    error("measure is not implemented for metric type $(typeof(metric))")
end

abstract type RankingMetric <: Metric end

function measure(metric::RankingMetric, truth::Array{T}, pred::Array{T}, k::Int) where T
    error("measure is not implemented for metric type $(typeof(metric))")
end

function count_true_positive(truth::Array{T}, pred::Array{T}) where T
    sum(in(truth), pred)
end

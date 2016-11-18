export Metric, AccuracyMetric, RankingMetric
export measure, count_true_positive

abstract Metric

abstract AccuracyMetric <: Metric

function measure(metric::AccuracyMetric, truth::AbstractVector, prediction::AbstractVector)
    error("measure is not implemented for metric type $(typeof(metric))")
end

abstract RankingMetric <: Metric

function measure{T}(metric::RankingMetric, truth::Array{T}, recommend::Array{T}, k::Int)
    error("measure is not implemented for metric type $(typeof(metric))")
end

function count_true_positive{T}(truth::Array{T}, recommend::Array{T})
    tp = 0
    for r in recommend
        if findfirst(truth, r) != 0
            tp += 1
        end
    end
    tp
end

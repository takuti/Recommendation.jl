export Recall, Precision, MAP, AUC, ReciprocalRank, MPR, NDCG

"""
    Recall

Recall@k.

    measure(
        metric::Recall,
        truth::Array{T},
        pred::Array{T},
        k::Int
    )
"""
struct Recall <: RankingMetric end
function measure(metric::Recall, truth::Array{T}, pred::Array{T}, k::Int) where T
    count_true_positive(truth, pred[1:k]) / length(truth)
end

"""
    Precision

Precision@k.

    measure(
        metric::Precision,
        truth::Array{T},
        pred::Array{T},
        k::Int
    )
"""
struct Precision <: RankingMetric end
function measure(metric::Precision, truth::Array{T}, pred::Array{T}, k::Int) where T
    count_true_positive(truth, pred[1:k]) / k
end

"""
    MAE

Mean Average Precision.

    measure(
        metric::MAP,
        truth::Array{T},
        pred::Array{T}
    )
"""
struct MAP <: RankingMetric end
function measure(metric::MAP, truth::Array{T}, pred::Array{T}, k::Int=0) where T
    tp = accum = 0
    n_pred = length(pred)

    for n = 1:n_pred
        if findfirst(isequal(pred[n]), truth) != nothing
            tp += 1
            accum += tp / n
        end
    end

    accum / length(truth)
end

"""
    AUC

Area Under the ROC Curve.

    measure(
        metric::AUC,
        truth::Array{T},
        pred::Array{T}
    )
"""
struct AUC <: RankingMetric end
function measure(metric::AUC, truth::Array{T}, pred::Array{T}, k::Int=0) where T
    tp = correct = 0
    for r in pred
        if findfirst(isequal(r), truth) != nothing
            # keep track number of tp placed before
            tp += 1
        else
            correct += tp
        end
    end
    # number of all possible tp-fp pairs
    pairs = tp * (length(pred) - tp)
    correct / pairs
end

"""
    ReciprocalRank

Reciprocal Rank.

    measure(
        metric::ReciprocalRank,
        truth::Array{T},
        pred::Array{T}
    )
"""
struct ReciprocalRank <: RankingMetric end
function measure(metric::ReciprocalRank, truth::Array{T}, pred::Array{T}, k::Int=0) where T
    n_pred = length(pred)
    for n = 1:n_pred
        if findfirst(isequal(pred[n]), truth) != nothing
            return 1 / n
        end
    end
    return 0
end

"""
    MPR

Mean Percentile Rank.

    measure(
        metric::MPR,
        truth::Array{T},
        pred::Array{T}
    )
"""
struct MPR <: RankingMetric end
function measure(metric::MPR, truth::Array{T}, pred::Array{T}, k::Int=0) where T
    accum = 0
    n_pred = length(pred)
    for t in truth
        r = (coalesce(findfirst(isequal(t), pred), 0) - 1) / n_pred
        accum += r
    end
    accum * 100 / length(truth)
end

"""
    NDCG

Normalized Discounted Cumulative Gain.

    measure(
        metric::NDCG,
        truth::Array{T},
        pred::Array{T},
        k::Int
    )
"""
struct NDCG <: RankingMetric end
function measure(metric::NDCG, truth::Array{T}, pred::Array{T}, k::Int) where T
    dcg = idcg = 0
    for n = 1:k
        d = 1 / log2(n + 1)
        if findfirst(isequal(pred[n]), truth) != nothing
            dcg += d
        end
        idcg += d
    end
    dcg / idcg
end

export Recall, Precision, MAP, AUC, ReciprocalRank, MPR, NDCG

# Recall@k
struct Recall <: RankingMetric end
function measure(metric::Recall, truth::Array{T}, pred::Array{T}, k::Int) where T
    count_true_positive(truth, pred[1:k]) / length(truth)
end

# Precision@k
struct Precision <: RankingMetric end
function measure(metric::Precision, truth::Array{T}, pred::Array{T}, k::Int) where T
    count_true_positive(truth, pred[1:k]) / k
end

# Mean Average Precision
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

# Area under the ROC curve
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

# Reciprocal Rank
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

# Mean Percentile Rank
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

# Normalized Discounted Cumulative Gain
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

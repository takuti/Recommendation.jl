export Recall, Precision, MAP, AUC, MRR, MPR, NDCG

# Recall@k
immutable Recall <: RankingMetric end
function measure{T}(metric::Recall, truth::Array{T}, rec::Array{T}, k::Int)
    count_true_positive(truth, rec[1:k]) / length(truth)
end

# Precision@k
immutable Precision <: RankingMetric end
function measure{T}(metric::Precision, truth::Array{T}, rec::Array{T}, k::Int)
    count_true_positive(truth, rec[1:k]) / k
end

# Mean Average Precision
immutable MAP <: RankingMetric end
function measure{T}(metric::MAP, truth::Array{T}, rec::Array{T}, k::Int=0)
    tp = accum = 0
    n_rec = length(rec)

    for n = 1:n_rec
        if findfirst(truth, rec[n]) != 0
            tp += 1
            accum += tp / n
        end
    end

    accum / length(truth)
end

# Area under the ROC curve
immutable AUC <: RankingMetric end
function measure{T}(metric::AUC, truth::Array{T}, rec::Array{T}, k::Int=0)
    tp = correct = 0
    for r in rec
        if findfirst(truth, r) != 0
            # keep track number of tp placed before
            tp += 1
        else
            correct += tp
        end
    end
    # number of all possible tp-fp pairs
    pairs = tp * (length(rec) - tp)
    correct / pairs
end

# Mean Reciprocal Rank
immutable MRR <: RankingMetric end
function measure{T}(metric::MRR, truth::Array{T}, rec::Array{T}, k::Int=0)
    n_rec = length(rec)
    for n = 1:n_rec
        if findfirst(truth, rec[n]) != 0
            return 1 / n
        end
    end
    return 0
end

# Mean Percentile Rank
immutable MPR <: RankingMetric end
function measure{T}(metric::MPR, truth::Array{T}, rec::Array{T}, k::Int=0)
    accum = 0
    n_rec = length(rec)
    for t in truth
        r = (findfirst(rec, t) - 1) / n_rec
        accum += r
    end
    accum * 100 / length(truth)
end

# Normalized Discounted Cumulative Gain
immutable NDCG <: RankingMetric end
function measure{T}(metric::NDCG, truth::Array{T}, rec::Array{T}, k::Int)
    dcg = idcg = 0
    for n = 1:k
        d = 1 / log2(n + 1)
        if findfirst(truth, rec[n]) != 0
            dcg += d
        end
        idcg += d
    end
    dcg / idcg
end

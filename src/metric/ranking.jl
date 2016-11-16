export RankingMetric

module RankingMetric

function count_true_positive{T}(truth::Array{T}, recommend::Array{T})
    tp = 0
    for r in recommend
        if findfirst(truth, r) != 0
            tp += 1
        end
    end
    tp
end

# Recall@k
function recall{T}(truth::Array{T}, recommend::Array{T}, k::Int)
    count_true_positive(truth, recommend[1:k]) / length(truth)
end

# Precision@k
function precision{T}(truth::Array{T}, recommend::Array{T}, k::Int)
    count_true_positive(truth, recommend[1:k]) / k
end

# Mean Average Precision
function map{T}(truth::Array{T}, recommend::Array{T}, k::Int=0)
    tp = accum = 0
    n_recommend = length(recommend)

    for n = 1:n_recommend
        if findfirst(truth, recommend[n]) != 0
            tp += 1
            accum += tp / n
        end
    end

    accum / length(truth)
end

# Area under the ROC curve
function auc{T}(truth::Array{T}, recommend::Array{T}, k::Int=0)
    tp = correct = 0
    for r in recommend
        if findfirst(truth, r) != 0
            # keep track number of tp placed before
            tp += 1
        else
            correct += tp
        end
    end
    # number of all possible tp-fp pairs
    pairs = tp * (length(recommend) - tp)
    correct / pairs
end

# Mean Reciprocal Rank
function mrr{T}(truth::Array{T}, recommend::Array{T}, k::Int=0)
    n_recommend = length(recommend)
    for n = 1:n_recommend
        if findfirst(truth, recommend[n]) != 0
            return 1 / n
        end
    end
    return 0
end

# Mean Percentile Rank
function mpr{T}(truth::Array{T}, recommend::Array{T}, k::Int=0)
    accum = 0
    n_recommend = length(recommend)
    for t in truth
        r = (findfirst(recommend, t) - 1) / n_recommend
        accum += r
    end
    accum * 100 / length(truth)
end

# Normalized Discounted Cumulative Gain
function ndcg{T}(truth::Array{T}, recommend::Array{T}, k::Int)
    dcg = idcg = 0
    for n = 1:k
        d = 1 / log2(n + 1)
        if findfirst(truth, recommend[n]) != 0
            dcg += d
        end
        idcg += d
    end
    dcg / idcg
end

end # module RankingMetric

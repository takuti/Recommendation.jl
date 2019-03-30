export Recall, Precision, MAP, AUC, ReciprocalRank, MPR, NDCG

"""
    Recall

Recall-at-``N`` (Recall@``N``) indicates coverage of truth samples as a result of top-``N`` recommendation. The value is computed by the following equation:
```math
\\mathrm{Recall@}N = \\frac{|\\mathcal{I}^+_u \\cap I_N(u)|}{|\\mathcal{I}^+_u|}.
```
Here, ``|\\mathcal{I}^+_u \\cap I_N(u)|`` is the number of *true positives*.

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

Precision-at-``N`` (Precision@``N``) evaluates correctness of a top-``N`` recommendation list ``I_N(u)`` according to the portion of true positives in the list as:
```math
\\mathrm{Precision@}N = \\frac{|\\mathcal{I}^+_u \\cap I_N(u)|}{|I_N(u)|}.
```

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
    MAP

While the original Precision@``N`` provides a score for a fixed-length recommendation list ``I_N(u)``, mean average precision (MAP) computes an average of the scores over all recommendation sizes from 1 to ``|\\mathcal{I}|``. MAP is formulated with an indicator function for ``i_n``, the ``n``-th item of ``I(u)``, as:
```math
\\mathrm{MAP} = \\frac{1}{|\\mathcal{I}^+_u|} \\sum_{n = 1}^{|\\mathcal{I}|} \\mathrm{Precision@}n \\cdot  \\mathbb{1}_{\\mathcal{I}^+_u}(i_n).
```

It should be noticed that, MAP is not a simple mean of sum of Precision@``1``, Precision@``2``, ``\\dots``, Precision@``|\\mathcal{I}|``, and higher-ranked true positives lead better MAP.

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

ROC curve and area under the ROC curve (AUC) are generally used in evaluation of the classification problems, but these concepts can also be interpreted in a context of ranking problem. Basically, the AUC metric for ranking considers all possible pairs of truth and other items which are respectively denoted by ``i^+ \\in \\mathcal{I}^+_u`` and ``i^- \\in \\mathcal{I}^-_u``, and it expects that the ``best'' recommender completely ranks ``i^+`` higher than ``i^-``, as follows:

![auc](./assets/images/auc.png)

AUC calculation keeps track the number of true positives at different rank in ``\\mathcal{I}``. At line 8, the function adds the number of true positives which were ranked higher than the current non-truth sample to the accumulated count of correct pairs. Ultimately, an AUC score is computed as portion of the correct ordered ``(i^+, i^-)`` pairs in the all possible combinations determined by ``|\\mathcal{I}^+_u| \\times |\\mathcal{I}^-_u|`` in set notation.

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

If we are only interested in the first true positive, reciprocal rank (RR) could be a reasonable choice to quantitatively assess the recommendation lists. For ``n_{\\mathrm{tp}} \\in \\left[ 1, |\\mathcal{I}| \\right]``, a position of the first true positive in ``I(u)``, RR simply returns its inverse:
```math
  \\mathrm{RR} = \\frac{1}{n_{\\mathrm{tp}}}.
```

RR can be zero if and only if ``\\mathcal{I}^+_u`` is empty.

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

Mean percentile rank (MPR) is a ranking metric based on ``r_{i} \\in [0, 100]``, the percentile-ranking of an item ``i`` within the sorted list of all items for a user ``u``. It can be formulated as:
```math
\\mathrm{MPR} = \\frac{1}{|\\mathcal{I}^+_u|} \\sum_{i \\in \\mathcal{I}^+_u} r_{i}.
```
``r_{i} = 0\\%`` is the best value that means the truth item ``i`` is ranked at the highest position in a recommendation list. On the other hand, ``r_{i} = 100\\%`` is the worst case that the item ``i`` is at the lowest rank.

MPR internally considers not only top-``N`` recommended items also all of the non-recommended items, and it accumulates the percentile ranks for all true positives unlike MRR. So, the measure is suitable to estimate users' overall satisfaction for a recommender. Intuitively, ``\\mathrm{MPR} > 50\\%`` should be worse than random ranking from a users' point of view.

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

Like MPR, normalized discounted cumulative gain (NDCG) computes a score for ``I(u)`` which places emphasis on higher-ranked true positives. In addition to being a more well-formulated measure, the difference between NDCG and MPR is that NDCG allows us to specify an expected ranking within ``\\mathcal{I}^+_u``; that is, the metric can incorporate ``\\mathrm{rel}_n``, a relevance score which suggests how likely the ``n``-th sample is to be ranked at the top of a recommendation list, and it directly corresponds to an expected ranking of the truth samples.

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

export Coverage, Novelty, IntraListSimilarity
export count_intersect

function count_intersect(s1::Union{AbstractSet, AbstractVector}, s2::Union{AbstractSet, AbstractVector})
    length(intersect(s1, s2))
end

struct Coverage <: IntraListMetric end
function measure(metric::Coverage, recommendations::Union{AbstractSet, AbstractVector}; catalog::Union{AbstractSet, AbstractVector})
    count_intersect(recommendations, catalog) / length(catalog)
end

struct Novelty <: IntraListMetric end
function measure(metric::Novelty, recommendations::Union{AbstractSet, AbstractVector}; observed::Union{AbstractSet, AbstractVector})
    # number of recommended items that have not been observed yet
    length(setdiff(recommendations, observed))
end

"""
http://files.grouplens.org/papers/ziegler-www05.pdf
"""
struct IntraListSimilarity <: IntraListMetric end
function measure(metric::IntraListSimilarity, recommendations::Union{AbstractSet, AbstractVector}; sims::AbstractMatrix)
    sum(map(t -> (t[1] == t[2]) ? 0.0 : sims[t...], Iterators.product(recommendations, recommendations))) / 2.0
end

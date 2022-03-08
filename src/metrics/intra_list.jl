export Coverage, Novelty, IntraListSimilarity, Serendipity
export count_intersect

"""
    count_intersect(s1::Union{AbstractSet, AbstractVector}, s2::Union{AbstractSet, AbstractVector}) -> Int

Count the number of elements that are in both `s1` and `s2`.
"""
function count_intersect(s1::Union{AbstractSet, AbstractVector}, s2::Union{AbstractSet, AbstractVector})
    length(intersect(s1, s2))
end

"""
    Coverage

Catalog coverage is a ratio of recommended items among `catalog`, which represents a set of all available items.

```julia
measure(
    metric::Coverage, recommendations::Union{AbstractSet, AbstractVector};
    catalog::Union{AbstractSet, AbstractVector}
)
```
"""
struct Coverage <: IntraListMetric end
function measure(metric::Coverage, recommendations::Union{AbstractSet, AbstractVector}; catalog::Union{AbstractSet, AbstractVector})
    count_intersect(recommendations, catalog) / length(catalog)
end

"""
    Novelty

The number of recommended items that have not been observed yet i.e., not in `observed`.

```julia
measure(
    metric::Novelty, recommendations::Union{AbstractSet, AbstractVector};
    observed::Union{AbstractSet, AbstractVector}
)
```
"""
struct Novelty <: IntraListMetric end
function measure(metric::Novelty, recommendations::Union{AbstractSet, AbstractVector}; observed::Union{AbstractSet, AbstractVector})
    # number of recommended items that have not been observed yet
    length(setdiff(recommendations, observed))
end

"""
    IntraListSimilarity

Sum of similarities between every pairs of recommended items. Larger value represents less diversity.

- Reference: [Improving Recommendation Lists Through Topic Diversification](http://files.grouplens.org/papers/ziegler-www05.pdf)

```julia
measure(
    metric::IntraListSimilarity, recommendations::Union{AbstractSet, AbstractVector};
    sims::AbstractMatrix
)
```
"""
struct IntraListSimilarity <: IntraListMetric end
function measure(metric::IntraListSimilarity, recommendations::Union{AbstractSet, AbstractVector}; sims::AbstractMatrix)
    sum(map(t -> (t[1] == t[2]) ? 0.0 : sims[t...], Iterators.product(recommendations, recommendations))) / 2.0
end

"""
    Serendipity

Return a sum of relevance-unexpectedness multiplications for all recommended items.

```julia
measure(
    metric::Serendipity, recommendations::Union{AbstractSet, AbstractVector};
    relevance::AbstractVector, unexpectedness::AbstractVector
)
```
"""
struct Serendipity <: IntraListMetric end
function measure(metric::Serendipity, recommendations::Union{AbstractSet, AbstractVector}; relevance::AbstractVector, unexpectedness::AbstractVector)
    sum(map(item -> relevance[item] * unexpectedness[item], recommendations))
end

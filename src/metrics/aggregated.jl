export AggregatedDiversity, ShannonEntropy, GiniIndex
export find_all_items, count_users_contain

"""
    find_all_items(recommendations::AbstractVector{<:AbstractVector{<:Integer}}) -> Set

Return a set of distinct items from an array of multiple top-k recommendation lists.
"""
function find_all_items(recommendations::AbstractVector{<:AbstractVector{<:Integer}})
    Set(reduce(vcat, recommendations))
end

"""
    AggregatedDiversity

The number of distinct items recommended across all suers. Larger value indicates more diverse recommendation result overall.

```julia
measure(
    metric::AggregatedDiversity, recommendations::AbstractVector{<:AbstractVector{<:Integer}}
)
```

Let ``U`` and ``I`` be a set of users and items, respectively, and ``L_N(u)`` a list of top-``N`` recommended items for a user ``u``. Here, an aggregated diversity can be calculated as:

```math
\\left| \\bigcup\\limits_{u \\in U} L_N(u) \\right|
```
"""
struct AggregatedDiversity <: AggregatedMetric end
function measure(metric::AggregatedDiversity, recommendations::AbstractVector{<:AbstractVector{<:Integer}})
    # number of distinct items recommended across all users
    length(find_all_items(recommendations))
end

"""
    count_users_contain(item::Integer, recommendations::AbstractVector{<:AbstractVector{<:Integer}}) -> Int

Given an array of top-k recommendation lists for multiple users, count the number of users who are recommended a particular item `item`.
"""
function count_users_contain(item::Integer, recommendations::AbstractVector{<:AbstractVector{<:Integer}})
    sum(map(recs -> (item in recs), recommendations))
end

"""
    ShannonEntropy

If we focus more on individual items and how many users are recommended a particular item, the diversity of top-`k` recommender can be defined by [Shannon Entropy](https://en.wikipedia.org/wiki/Entropy_(information_theory)):

```math
-\\sum_{j = 1}^{|I|} \\left( \\frac{\\left|\\{u \\mid u \\in U \\wedge i_j \\in L_N(u) \\}\\right|}{N |U|} \\ln \\left( \\frac{\\left|\\{u \\mid u \\in U \\wedge i_j \\in L_N(u) \\}\\right|}{N |U|}  \\right) \\right)
```
where ``i_j`` denotes ``j``-th item in the available item set ``I``.

```julia
measure(
    metric::ShannonEntropy, recommendations::AbstractVector{<:AbstractVector{<:Integer}};
    k::Integer
)
```
"""
struct ShannonEntropy <: AggregatedMetric end
function measure(metric::ShannonEntropy, recommendations::AbstractVector{<:AbstractVector{<:Integer}}; k::Integer)
    n_users = size(recommendations, 1)
    items = find_all_items(recommendations)
    entropy = 0
    for item in items
        p_i = count_users_contain(item, recommendations) / (k * n_users)
        entropy += p_i * log(p_i)
    end
    -entropy
end

"""
    GiniIndex

[Gini Index](https://en.wikipedia.org/wiki/Gini_coefficient), which is normally used to measure a degree of inequality in a distribution of income, can be applied to assess diversity in the context of top-`k` recommendation:

```math
\\frac{1}{|I| - 1} \\sum_{j = 1}^{|I|} \\left( (2j - |I| - 1) \\cdot \\frac{\\left|\\{u \\mid u \\in U \\wedge i_j \\in L_N(u) \\}\\right|}{N |U|} \\right)
```

```julia
measure(
    metric::GiniIndex, recommendations::AbstractVector{<:AbstractVector{<:Integer}};
    k::Integer
)
```

The index is 0 when all items are equally chosen in terms of the number of recommended users.
"""
struct GiniIndex <: AggregatedMetric end
function measure(metric::GiniIndex, recommendations::AbstractVector{<:AbstractVector{<:Integer}}; k::Integer)
    n_users = size(recommendations, 1)
    probs = sort(map(item -> count_users_contain(item, recommendations) / (k * n_users), collect(find_all_items(recommendations))))
    if first(probs) == last(probs) #  all items are chosen equally often
        return 0.0
    end
    n_items = length(probs)
    gini = 0
    for (index, p_i) in enumerate(probs)
        gini += (2 * index - n_items - 1) * p_i
    end
    gini / (n_items - 1)
end

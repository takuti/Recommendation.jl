export Metric, AccuracyMetric, RankingMetric
export measure, count_intersect, coverage, find_all_items, aggregated_diversity, novelty, count_users_contain, entropy, gini

abstract type Metric end

abstract type AccuracyMetric <: Metric end

function measure(metric::AccuracyMetric, truth::AbstractVector, pred::AbstractVector)
    error("measure is not implemented for metric type $(typeof(metric))")
end

abstract type RankingMetric <: Metric end

function measure(metric::RankingMetric, truth::AbstractVector{T}, pred::AbstractVector{T}, k::Integer) where T
    error("measure is not implemented for metric type $(typeof(metric))")
end

function count_intersect(s1::Union{AbstractSet, AbstractVector}, s2::Union{AbstractSet, AbstractVector})
    length(intersect(s1, s2))
end

function coverage(items::Union{AbstractSet, AbstractVector}, catalog::Union{AbstractSet, AbstractVector})
    count_intersect(items, catalog) / length(catalog)
end

function find_all_items(recommendations::AbstractVector{<:AbstractVector{<:Integer}})
    Set(reduce(vcat, recommendations))
end

function aggregated_diversity(recommendations::AbstractVector{<:AbstractVector{<:Integer}})
    # number of distinct items recommended across all users
    length(find_all_items(recommendations))
end

function novelty(recommendations::AbstractVector{<:AbstractVector{<:Integer}}, observed::AbstractVector{<:AbstractVector{<:Integer}})
    # avg number of recommended items that have not been observed yet
    mean(map(t -> length(setdiff(t[1], t[2])), zip(recommendations, observed)))
end

function count_users_contain(item::Integer, recommendations::AbstractVector{<:AbstractVector{<:Integer}})
    sum(map(recs -> (item in recs), recommendations))
end

function entropy(recommendations::AbstractVector{<:AbstractVector{<:Integer}}, n::Integer)
    n_users = size(recommendations, 1)
    items = find_all_items(recommendations)
    entropy = 0
    for item in items
        p_i = count_users_contain(item, recommendations) / (n * n_users)
        entropy += p_i * log(p_i)
    end
    -entropy
end

function gini(recommendations::AbstractVector{<:AbstractVector{<:Integer}}, n::Integer)
    n_users = size(recommendations, 1)
    probs = sort(map(item -> count_users_contain(item, recommendations) / (n * n_users), collect(find_all_items(recommendations))))
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

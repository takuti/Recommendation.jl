export AggregatedDiversity, ShannonEntropy, GiniIndex
export find_all_items, count_users_contain

function find_all_items(recommendations::AbstractVector{<:AbstractVector{<:Integer}})
    Set(reduce(vcat, recommendations))
end

struct AggregatedDiversity <: AggregatedMetric end
function measure(metric::AggregatedDiversity, recommendations::AbstractVector{<:AbstractVector{<:Integer}})
    # number of distinct items recommended across all users
    length(find_all_items(recommendations))
end

function count_users_contain(item::Integer, recommendations::AbstractVector{<:AbstractVector{<:Integer}})
    sum(map(recs -> (item in recs), recommendations))
end

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

export ItemMean

"""
    ItemMean(data::DataAccessor)

Recommend based on global item mean rating.
"""
struct ItemMean <: Recommender
    data::DataAccessor
    scores::AbstractVector
    states::States

    function ItemMean(data::DataAccessor)
        n_item = size(data.R, 2)
        new(data, zeros(n_item), States(:built => false))
    end
end

function build!(recommender::ItemMean)
    n_item = size(recommender.data.R, 2)

    for i in 1:n_item
        v = recommender.data.R[:, i]
        recommender.scores[i] = sum(v) / count(!iszero, v)
    end

    recommender.states[:built] = true
end

function predict(recommender::ItemMean, u::Int, i::Int)
    check_build_status(recommender)
    recommender.scores[i]
end

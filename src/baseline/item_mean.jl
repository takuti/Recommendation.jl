export ItemMean

"""
    ItemMean(da::DataAccessor)

Recommend based on global item mean rating.
"""
struct ItemMean <: Recommender
    da::DataAccessor
    scores::AbstractVector
    states::States

    function ItemMean(da::DataAccessor)
        n_item = size(da.R, 2)
        new(da, zeros(n_item), States(:built => false))
    end
end

function build!(recommender::ItemMean)
    n_item = size(recommender.da.R, 2)

    for i in 1:n_item
        v = recommender.da.R[:, i]
        recommender.scores[i] = sum(v) / count(!iszero, v)
    end

    recommender.states[:built] = true
end

function predict(recommender::ItemMean, u::Int, i::Int)
    check_build_status(recommender)
    recommender.scores[i]
end

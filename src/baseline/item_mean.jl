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

function build(rec::ItemMean)
    n_item = size(rec.da.R, 2)

    for i in 1:n_item
        v = rec.da.R[:, i]
        rec.scores[i] = sum(v) / count(!iszero, v)
    end

    rec.states[:built] = true
end

function predict(rec::ItemMean, u::Int, i::Int)
    check_build_status(rec)
    rec.scores[i]
end

export ItemMean

struct ItemMean <: Recommender
    da::DataAccessor
    scores::AbstractVector
    states::States
end

"""
    ItemMean(da::DataAccessor)

Recommend based on global item mean rating.
"""
ItemMean(da::DataAccessor, hyperparams::Parameters=Parameters()) = begin
    n_item = size(da.R, 2)
    ItemMean(da, zeros(n_item), States(:is_built => false))
end

function build(rec::ItemMean)
    n_item = size(rec.da.R, 2)

    for i in 1:n_item
        v = rec.da.R[:, i]
        rec.scores[i] = sum(v) / count(!iszero, v)
    end

    rec.states[:is_built] = true
end

function predict(rec::ItemMean, u::Int, i::Int)
    check_build_status(rec)
    rec.scores[i]
end

export ItemMean

immutable ItemMean <: Recommender
    da::DataAccessor
    scores::AbstractVector
end

ItemMean(da::DataAccessor) = begin
    n_item = size(da.R, 2)
    ItemMean(da, zeros(n_item))
end

function build(rec::ItemMean)
    n_item = size(rec.da.R, 2)

    for i in 1:n_item
        v = rec.da.R[:, i]
        rec.scores[i] = sum(v) / countnz(v)
    end
end

function predict(rec::ItemMean, u::Int, i::Int)
    rec.scores[i]
end

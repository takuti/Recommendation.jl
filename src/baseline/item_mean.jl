export ItemMean

immutable ItemMean <: Recommender
    da::DataAccessor
    scores::AbstractVector
end

ItemMean(da::DataAccessor) = begin
    n_item = size(da.R, 2)
    ItemMean(da, zeros(n_item))
end

function build(recommender::ItemMean)
    n_item = size(recommender.da.R, 2)

    for i in 1:n_item
        v = recommender.da.R[:, i]
        recommender.scores[i] = sum(v) / countnz(v)
    end
end

function predict(recommender::ItemMean, u::Int, i::Int)
    recommender.scores[i]
end

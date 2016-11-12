export ItemMean

immutable ItemMean <: Recommender
    da::DataAccessor
    scores::AbstractVector
end

ItemMean(da::DataAccessor) = begin
    n_user, n_item = size(da.R)

    scores = zeros(n_item)

    for i in 1:n_item
        v = da.R[:, i]
        scores[i] = sum(v) / countnz(v)
    end

    ItemMean(da, scores)
end

function predict(recommender::ItemMean, u::Int, i::Int)
    recommender.scores[i]
end

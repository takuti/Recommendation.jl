export ItemMean

immutable ItemMean <: Recommender
    m::AbstractMatrix
    scores::AbstractVector
end

ItemMean(m::AbstractMatrix) = begin
    n_user, n_item = size(m)

    scores = zeros(n_item)

    for i in 1:n_item
        v = m[:, i]
        scores[i] = sum(v) / countnz(v)
    end

    ItemMean(m, scores)
end

function predict(recommender::ItemMean, u::Int, i::Int)
    recommender.scores[i]
end

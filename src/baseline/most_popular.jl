export MostPopular

immutable MostPopular <: Recommender
    m::AbstractMatrix
    scores::AbstractVector
end

MostPopular(m::AbstractMatrix) = begin
    n_user, n_item = size(m)

    scores = zeros(n_item)

    for i in 1:n_item
        scores[i] = countnz(m[:, i])
    end

    MostPopular(m, scores)
end

function ranking(recommender::MostPopular, u::Int, i::Int)
    recommender.scores[i]
end

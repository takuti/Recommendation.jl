export MostPopular

immutable MostPopular <: Recommender
    da::DataAccessor
    scores::AbstractVector
end

MostPopular(da::DataAccessor) = begin
    n_user, n_item = size(da.R)

    scores = zeros(n_item)

    for i in 1:n_item
        scores[i] = countnz(da.R[:, i])
    end

    MostPopular(da, scores)
end

function ranking(recommender::MostPopular, u::Int, i::Int)
    recommender.scores[i]
end

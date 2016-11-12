export MostPopular

immutable MostPopular <: Recommender
    da::DataAccessor
    scores::AbstractVector
end

MostPopular(da::DataAccessor) = begin
    n_item = size(da.R, 2)
    MostPopular(da, zeros(n_item))
end

function build(recommender::MostPopular)
    n_item = size(recommender.da.R, 2)

    for i in 1:n_item
        recommender.scores[i] = countnz(recommender.da.R[:, i])
    end
end

function ranking(recommender::MostPopular, u::Int, i::Int)
    recommender.scores[i]
end

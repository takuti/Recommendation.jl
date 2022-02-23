export ItemMean

"""
    ItemMean(data::DataAccessor)

Recommend based on global item mean rating.
"""
struct ItemMean <: Recommender
    data::DataAccessor
    scores::AbstractVector

    function ItemMean(data::DataAccessor)
        n_item = size(data.R, 2)
        new(data, vector(n_item))
    end
end

isdefined(recommender::ItemMean) = isfilled(recommender.scores)

function fit!(recommender::ItemMean)
    recommender.scores[:] = vec(mean(recommender.data.R, dims=1))
end

function predict(recommender::ItemMean, u::Integer, i::Integer)
    validate(recommender)
    recommender.scores[i]
end

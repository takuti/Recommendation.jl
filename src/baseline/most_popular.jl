export MostPopular

"""

    MostPopular(data::DataAccessor)

Recommend most popular items.
"""
struct MostPopular <: Recommender
    data::DataAccessor
    scores::AbstractVector

    function MostPopular(data::DataAccessor)
        n_item = size(data.R, 2)
        new(data, vector(n_item))
    end
end

isdefined(recommender::MostPopular) = isfilled(recommender.scores)

function fit!(recommender::MostPopular)
    recommender.scores[:] = vec(sum(!iszero, recommender.data.R, dims=1))
end

function ranking(recommender::MostPopular, u::Integer, i::Integer)
    validate(recommender)
    recommender.scores[i]
end

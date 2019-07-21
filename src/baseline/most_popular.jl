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

isbuilt(recommender::MostPopular) = isfilled(recommender.scores)

function build!(recommender::MostPopular)
    n_item = size(recommender.data.R, 2)

    for i in 1:n_item
        recommender.scores[i] = count(!iszero, recommender.data.R[:, i])
    end
end

function ranking(recommender::MostPopular, u::Int, i::Int)
    check_build_status(recommender)
    recommender.scores[i]
end

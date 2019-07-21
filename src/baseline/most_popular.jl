export MostPopular

"""

    MostPopular(da::DataAccessor)

Recommend most popular items.
"""
struct MostPopular <: Recommender
    da::DataAccessor
    scores::AbstractVector
    states::States

    function MostPopular(da::DataAccessor)
        n_item = size(da.R, 2)
        new(da, zeros(n_item), States(:built => false))
    end
end

function build!(recommender::MostPopular)
    n_item = size(recommender.da.R, 2)

    for i in 1:n_item
        recommender.scores[i] = count(!iszero, recommender.da.R[:, i])
    end

    recommender.states[:built] = true
end

function ranking(recommender::MostPopular, u::Int, i::Int)
    check_build_status(recommender)
    recommender.scores[i]
end

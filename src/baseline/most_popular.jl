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
        new(da, zeros(n_item), States(:is_built => false))
    end
end

function build(rec::MostPopular)
    n_item = size(rec.da.R, 2)

    for i in 1:n_item
        rec.scores[i] = count(!iszero, rec.da.R[:, i])
    end

    rec.states[:is_built] = true
end

function ranking(rec::MostPopular, u::Int, i::Int)
    check_build_status(rec)
    rec.scores[i]
end

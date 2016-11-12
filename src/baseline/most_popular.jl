export MostPopular

immutable MostPopular <: Recommender
    da::DataAccessor
    scores::AbstractVector
end

MostPopular(da::DataAccessor) = begin
    n_item = size(da.R, 2)
    MostPopular(da, zeros(n_item))
end

function build(rec::MostPopular)
    n_item = size(rec.da.R, 2)

    for i in 1:n_item
        rec.scores[i] = countnz(rec.da.R[:, i])
    end
end

function ranking(rec::MostPopular, u::Int, i::Int)
    rec.scores[i]
end

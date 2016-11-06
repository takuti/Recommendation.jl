export MostPopular

immutable MostPopular <: Recommender
    m::AbstractMatrix
end

function ranking(recommender::MostPopular, u::Int, i::Int)
    countnz(recommender.m[:, i])
end

export MostPopular

immutable MostPopular <: Recommender
    mat::Array{Float64,2}
end

function ranking(recommender::MostPopular, u::Int, i::Int)
    countnz(recommender.mat[:, i])
end

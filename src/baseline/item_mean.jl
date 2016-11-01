export ItemMean

immutable ItemMean <: Recommender
    mat::Array{Float64,2}
end

function predict(recommender::ItemMean, u::Int, i::Int)
    mean(recommender.mat[:, i])
end

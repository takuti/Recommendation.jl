export UserMean

immutable UserMean <: Recommender
    mat::Array{Float64,2}
end

function predict(recommender::UserMean, u::Int, i::Int)
    mean(recommender.mat[u, :])
end

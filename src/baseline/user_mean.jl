export UserMean

immutable UserMean <: Recommender
    da::DataAccessor
    scores::AbstractVector
end

UserMean(da::DataAccessor) = begin
    n_user = size(da.R, 1)
    UserMean(da, zeros(n_user))
end

function build(recommender::UserMean)
    n_user = size(recommender.da.R, 1)

    for u in 1:n_user
        v = recommender.da.R[u, :]
        recommender.scores[u] = sum(v) / countnz(v)
    end
end

function predict(recommender::UserMean, u::Int, i::Int)
    recommender.scores[u]
end

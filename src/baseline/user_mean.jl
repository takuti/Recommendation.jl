export UserMean

immutable UserMean <: Recommender
    da::DataAccessor
    scores::AbstractVector
end

UserMean(da::DataAccessor) = begin
    n_user, n_item = size(da.R)

    scores = zeros(n_user)

    for u in 1:n_user
        v = da.R[u, :]
        scores[u] = sum(v) / countnz(v)
    end

    UserMean(da, scores)
end

function predict(recommender::UserMean, u::Int, i::Int)
    recommender.scores[u]
end

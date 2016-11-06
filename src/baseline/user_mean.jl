export UserMean

immutable UserMean <: Recommender
    m::AbstractMatrix
    scores::AbstractVector
end

UserMean(m::AbstractMatrix) = begin
    n_user, n_item = size(m)

    scores = zeros(n_user)

    for u in 1:n_user
        v = m[u, :]
        scores[u] = sum(v) / countnz(v)
    end

    UserMean(m, scores)
end

function predict(recommender::UserMean, u::Int, i::Int)
    recommender.scores[u]
end

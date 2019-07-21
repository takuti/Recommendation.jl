export UserMean

"""
    UserMean(da::DataAccessor)

Recommend based on global user mean rating.
"""
struct UserMean <: Recommender
    da::DataAccessor
    scores::AbstractVector
    states::States

    function UserMean(da::DataAccessor)
        n_user = size(da.R, 1)
        new(da, zeros(n_user), States(:built => false))
    end
end

function build!(recommender::UserMean)
    n_user = size(recommender.da.R, 1)

    for u in 1:n_user
        v = recommender.da.R[u, :]
        recommender.scores[u] = sum(v) / count(!iszero, v)
    end

    recommender.states[:built] = true
end

function predict(recommender::UserMean, u::Int, i::Int)
    check_build_status(recommender)
    recommender.scores[u]
end

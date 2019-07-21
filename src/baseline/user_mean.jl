export UserMean

"""
    UserMean(data::DataAccessor)

Recommend based on global user mean rating.
"""
struct UserMean <: Recommender
    data::DataAccessor
    scores::AbstractVector

    function UserMean(data::DataAccessor)
        n_user = size(data.R, 1)
        new(data, vector(n_user))
    end
end

isbuilt(recommender::UserMean) = isfilled(recommender.scores)

function build!(recommender::UserMean)
    n_user = size(recommender.data.R, 1)

    for u in 1:n_user
        v = recommender.data.R[u, :]
        recommender.scores[u] = sum(v) / count(!iszero, v)
    end
end

function predict(recommender::UserMean, u::Int, i::Int)
    check_build_status(recommender)
    recommender.scores[u]
end

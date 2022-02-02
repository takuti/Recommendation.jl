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

isdefined(recommender::UserMean) = isfilled(recommender.scores)

function fit!(recommender::UserMean)
    n_user = size(recommender.data.R, 1)

    for u in 1:n_user
        v = recommender.data.R[u, :]
        recommender.scores[u] = mean(v)
    end
end

function predict(recommender::UserMean, u::Integer, i::Integer)
    validate(recommender)
    recommender.scores[u]
end

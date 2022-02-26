export UserMean

"""
    UserMean(data::DataAccessor)

Recommend based on global user mean rating.
"""
struct UserMean <: Recommender
    data::DataAccessor
    scores::AbstractVector

    function UserMean(data::DataAccessor)
        n_users = size(data.R, 1)
        new(data, vector(n_users))
    end
end

isdefined(recommender::UserMean) = isfilled(recommender.scores)

function fit!(recommender::UserMean)
    # equivalent to vec(mean(recommender.data.R, dims=2)),
    # but avoid using `mean` as `dims=2` shows poor performance
    recommender.scores[:] = vec(sum(recommender.data.R, dims=2)) / size(recommender.data.R, 2)
end

function predict(recommender::UserMean, u::Integer, i::Integer)
    validate(recommender)
    recommender.scores[u]
end

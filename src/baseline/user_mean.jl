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
        new(da, zeros(n_user), States(:is_built => false))
    end
end

function build(rec::UserMean)
    n_user = size(rec.da.R, 1)

    for u in 1:n_user
        v = rec.da.R[u, :]
        rec.scores[u] = sum(v) / count(!iszero, v)
    end

    rec.states[:is_built] = true
end

function predict(rec::UserMean, u::Int, i::Int)
    check_build_status(rec)
    rec.scores[u]
end

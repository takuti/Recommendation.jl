export UserMean

immutable UserMean <: Recommender
    da::DataAccessor
    scores::AbstractVector
end

UserMean(da::DataAccessor) = begin
    n_user = size(da.R, 1)
    UserMean(da, zeros(n_user))
end

function build(rec::UserMean)
    n_user = size(rec.da.R, 1)

    for u in 1:n_user
        v = rec.da.R[u, :]
        rec.scores[u] = sum(v) / countnz(v)
    end
end

function predict(rec::UserMean, u::Int, i::Int)
    rec.scores[u]
end

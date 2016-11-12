export SVD

immutable SVD <: Recommender
    da::DataAccessor
    U::AbstractMatrix
    S::AbstractVector
    V::AbstractMatrix
    k::Int
end

SVD(da::DataAccessor, k::Int) = begin
    # NaNs are filled by zeros for now
    da.R[isnan(da.R)] = 0

    res = svds(da.R, nsv=k)[1]
    SVD(da, res.U, res.S, res.Vt, k)
end

function predict(recommender::SVD, u::Int, i::Int)
    dot(recommender.U[u, :] .* recommender.S, recommender.V[i, :])
end

export SVD

immutable SVD <: Recommender
    da::DataAccessor
    R_approx::AbstractMatrix
    U::AbstractMatrix
    S::AbstractVector
    V::AbstractMatrix
    k::Int
end

SVD(da::DataAccessor, k::Int) = begin
    # NaNs are filled by zeros for now
    da.R[isnan(da.R)] = 0

    res = svds(da.R, nsv=k)[1]
    R_approx = res.U * diagm(res.S) * res.Vt'

    SVD(da, R_approx, res.U, res.S, res.Vt, k)
end

function predict(recommender::SVD, u::Int, i::Int)
    recommender.R_approx[u, i]
end

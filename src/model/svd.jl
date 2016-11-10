export SVD

immutable SVD <: Recommender
    m::AbstractMatrix
    m_approx::AbstractMatrix
    U::AbstractMatrix
    S::AbstractVector
    V::AbstractMatrix
    k::Int
end

SVD(m::AbstractMatrix, k::Int) = begin
    # NaNs are filled by zeros for now
    m[isnan(m)] = 0

    res = svds(m, nsv=k)[1]
    m_approx = res.U * diagm(res.S) * res.Vt'

    SVD(m, m_approx, res.U, res.S, res.Vt, k)
end

function predict(recommender::SVD, u::Int, i::Int)
    recommender.m_approx[u, i]
end

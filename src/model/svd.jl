export SVD

"""
    SVD(
        data::DataAccessor,
        n_factors::Integer
    )

Recommendation based on SVD of a user-item matrix ``R \\in \\mathbb{R}^{|\\mathcal{U}| \\times |\\mathcal{I}|}``, which was originally studied by [Sarwar et al.](http://files.grouplens.org/papers/webKDD00.pdf) Rank ``k`` is configured by `n_factors`. Sparse matrix is not supported for `data.R`.

In a context of recommendation, ``U_k \\in \\mathbb{R}^{|\\mathcal{U}| \\times k}``, ``V \\in \\mathbb{R}^{|\\mathcal{I}| \\times k}`` and ``\\Sigma \\in \\mathbb{R}^{k \\times k}`` are respectively seen as ``k`` user/item feature vectors and corresponding weights. The idea of low-rank approximation that discards lower singular values intuitively works as *compression* or *denoising* of the original matrix; that is, each element in a rank-``k`` matrix ``A_k`` holds the best *compressed* (or *denoised*) value of the original element in ``A``. Thus, ``R_k = \\mathrm{SVD}_k(R)``, the best rank-``k`` approximation of ``R``, captures as much as possible of underlying users' preferences. Once ``R`` is decomposed into ``U, \\Sigma`` and ``V``, a ``(u, i)`` element of ``R_k`` calculated by ``\\sum^k_{j=1} \\sigma_j u_{u, j} v_{i, j}`` could be a prediction for the user-item pair.
"""
struct SVD <: Recommender
    data::DataAccessor
    n_factors::Integer
    U::AbstractMatrix
    S::AbstractVector
    Vt::AbstractMatrix

    function SVD(data::DataAccessor, n_factors::Integer)
        n_users, n_items = size(data.R)
        U = matrix(n_users, n_factors)
        S = vector(n_factors)
        Vt = matrix(n_factors, n_items)
        new(data, n_factors, U, S, Vt)
    end
end

SVD(data::DataAccessor) = SVD(data, 20)

isdefined(recommender::SVD) = isfilled(recommender.U)

function fit!(recommender::SVD)
    res = svd(recommender.data.R)
    recommender.U[:] = res.U[:, 1:recommender.n_factors]
    recommender.S[:] = res.S[1:recommender.n_factors]
    recommender.Vt[:] = res.Vt[1:recommender.n_factors, :]
end

function predict(recommender::SVD, u::Integer, i::Integer)
    validate(recommender)
    dot(recommender.U[u, :] .* recommender.S, recommender.Vt[:, i])
end

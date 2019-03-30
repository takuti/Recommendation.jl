export SVD

struct SVD <: Recommender
    da::DataAccessor
    hyperparams::Parameters
    params::Parameters
    states::States
end

"""
    SVD(
        da::DataAccessor,
        hyperparams::Parameters=Parameters(:k => 20)
    )

Recommendation based on SVD of a user-item matrix ``R \\in \\mathbb{R}^{|\\mathcal{U}| \\times |\\mathcal{I}|}``, which was originally studied by [Sarwar et al.](http://files.grouplens.org/papers/webKDD00.pdf) Rank ``k`` is configured by `k`.

In a context of recommendation, ``U_k \\in \\mathbb{R}^{|\\mathcal{U}| \\times k}``, ``V \\in \\mathbb{R}^{|\\mathcal{I}| \\times k}`` and ``\\Sigma \\in \\mathbb{R}^{k \\times k}`` are respectively seen as ``k`` user/item feature vectors and corresponding weights. The idea of low-rank approximation that discards lower singular values intuitively works as *compression* or *denoising* of the original matrix; that is, each element in a rank-``k`` matrix ``A_k`` holds the best *compressed* (or *denoised*) value of the original element in ``A``. Thus, ``R_k = \\mathrm{SVD}_k(R)``, the best rank-``k`` approximation of ``R``, captures as much as possible of underlying users' preferences. Once ``R`` is decomposed into ``U, \\Sigma`` and ``V``, a ``(u, i)`` element of ``R_k`` calculated by ``\\sum^k_{j=1} \\sigma_j u_{u, j} v_{i, j}`` could be a prediction for the user-item pair.
"""
SVD(da::DataAccessor,
    hyperparams::Parameters=Parameters(:k => 20)) = begin
    n_user, n_item = size(da.R)
    params = Parameters(:U => zeros(n_user, hyperparams[:k]),
                        :S => zeros(hyperparams[:k]),
                        :V => zeros(n_item, hyperparams[:k]))
    SVD(da, hyperparams, params, States(:is_built => false))
end

function build(rec::SVD)
    # NaNs are filled by zeros for now
    R = copy(rec.da.R)
    R[isnan.(R)] .= 0

    res = svd(R)
    rec.params[:U] = res.U[:, 1:rec.hyperparams[:k]]
    rec.params[:S] = res.S[1:rec.hyperparams[:k]]
    rec.params[:Vt] = res.Vt[1:rec.hyperparams[:k], :]

    rec.states[:is_built] = true
end

function predict(rec::SVD, u::Int, i::Int)
    check_build_status(rec)
    dot(rec.params[:U][u, :] .* rec.params[:S], rec.params[:Vt][:, i])
end

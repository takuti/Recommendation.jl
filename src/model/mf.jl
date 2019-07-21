export MF

"""
    MF(
        data::DataAccessor,
        k::Int
    )

Recommendation based on matrix factorization (MF). Number of factors is configured by `k`.

MF solves the following minimization problem for a set of observed user-item interactions ``\\mathcal{S} = \\{(u, i) \\in \\mathcal{U} \\times \\mathcal{I}\\}``:

```math
\\min_{P, Q} \\sum_{(u, i) \\in \\mathcal{S}} \\left( r_{u,i} - \\mathbf{p}_u^{\\mathrm{T}} \\mathbf{q}_i \\right)^2 + \\lambda \\ (\\|\\mathbf{p}_u\\|^2 + \\|\\mathbf{q}_i\\|^2),
```

where ``\\mathbf{p}_u, \\mathbf{q}_i \\in \\mathbb{R}^k`` are respectively a factorized user and item vector, and ``\\lambda`` is a regularization parameter to avoid overfitting. An optimal solution will be found by stochastic gradient descent (SGD). Ultimately, we can predict missing values in ``R`` by just computing ``PQ^{\\mathrm{T}}``, and the prediction directly leads recommendation.
"""
struct MF <: Recommender
    data::DataAccessor
    k::Int
    P::AbstractMatrix
    Q::AbstractMatrix

    function MF(data::DataAccessor, k::Int)
        n_user, n_item = size(data.R)
        P = matrix(n_user, k)
        Q = matrix(n_item, k)

        new(data, k, P, Q)
    end
end

MF(data::DataAccessor) = MF(data, 20)

isbuilt(recommender::MF) = isfilled(recommender.P)

function build!(recommender::MF;
               reg::Float64=1e-3, learning_rate::Float64=1e-3,
               eps::Float64=1e-3, max_iter::Int=100)
    n_user, n_item = size(recommender.data.R)

    # initialize with small values
    # (random is also possible)
    P = ones(n_user, recommender.k) * 0.1
    Q = ones(n_item, recommender.k) * 0.1

    pairs = vec([(u, i) for u in 1:n_user, i in 1:n_item])
    for it in 1:max_iter
        converged = true

        shuffled_pairs = shuffle(pairs)
        for (u, i) in shuffled_pairs
            r = recommender.data.R[u, i]
            if isnan(r); continue; end

            uv, iv = P[u, :], Q[i, :]

            err = r - dot(uv, iv)
            if abs(err) >= eps; converged = false; end

            grad = -2 * (err * iv - reg * uv)
            P[u, :] = uv - learning_rate * grad

            grad = -2 * (err * uv - reg * iv)
            Q[i, :] = iv - learning_rate * grad
        end

        if converged; break; end;
    end

    recommender.P[:] = P[:]
    recommender.Q[:] = Q[:]
end

function predict(recommender::MF, u::Int, i::Int)
    check_build_status(recommender)
    dot(recommender.P[u, :], recommender.Q[i, :])
end

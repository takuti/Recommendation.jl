export MatrixFactorization, MF

"""
    MatrixFactorization(
        data::DataAccessor,
        k::Integer
    )

Recommendation based on matrix factorization (MF). Number of factors is configured by `k`.

MF solves the following minimization problem for a set of observed user-item interactions ``\\mathcal{S} = \\{(u, i) \\in \\mathcal{U} \\times \\mathcal{I}\\}``:

```math
\\min_{P, Q} \\sum_{(u, i) \\in \\mathcal{S}} \\left( r_{u,i} - \\mathbf{p}_u^{\\mathrm{T}} \\mathbf{q}_i \\right)^2 + \\lambda \\ (\\|\\mathbf{p}_u\\|^2 + \\|\\mathbf{q}_i\\|^2),
```

where ``\\mathbf{p}_u, \\mathbf{q}_i \\in \\mathbb{R}^k`` are respectively a factorized user and item vector, and ``\\lambda`` is a regularization parameter to avoid overfitting. An optimal solution will be found by stochastic gradient descent (SGD). Ultimately, we can predict missing values in ``R`` by just computing ``PQ^{\\mathrm{T}}``, and the prediction directly leads recommendation.
"""
struct MatrixFactorization <: Recommender
    data::DataAccessor
    k::Integer
    P::AbstractMatrix
    Q::AbstractMatrix

    function MatrixFactorization(data::DataAccessor, k::Integer)
        n_user, n_item = size(data.R)
        P = matrix(n_user, k)
        Q = matrix(n_item, k)

        new(data, k, P, Q)
    end
end

"""
    MF(
        data::DataAccessor,
        k::Integer
    )

Alias of `MatrixFactorization`.
"""
const MF = MatrixFactorization

MF(data::DataAccessor) = MF(data, 20)

isdefined(recommender::MF) = isfilled(recommender.P)

function fit!(recommender::MF;
               reg::Float64=1e-3, learning_rate::Float64=1e-3,
               eps::Float64=1e-3, max_iter::Int=100,
               random_init::Bool=false)
    n_user, n_item = size(recommender.data.R)

    if random_init
        P = rand(Float64, size(recommender.P))
        Q = rand(Float64, size(recommender.Q))
    else
        # initialize with small constants
        P = ones(size(recommender.P)) * 0.1
        Q = ones(size(recommender.Q)) * 0.1
    end

    pairs = vec([(u, i) for u in 1:n_user, i in 1:n_item])
    for it in 1:max_iter
        converged = true

        shuffled_pairs = shuffle(pairs)
        for (u, i) in shuffled_pairs
            r = recommender.data.R[u, i]
            if iszero(r); continue; end

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

function predict(recommender::MF, u::Integer, i::Integer)
    validate(recommender)
    dot(recommender.P[u, :], recommender.Q[i, :])
end

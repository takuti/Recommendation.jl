export MatrixFactorization, MF

"""
    MatrixFactorization(
        data::DataAccessor,
        n_factors::Integer
    )

Recommendation based on matrix factorization (MF). Number of factors ``k`` is configured by `n_factors`.

MF solves the following minimization problem for a set of observed user-item interactions ``\\mathcal{S} = \\{(u, i) \\in \\mathcal{U} \\times \\mathcal{I}\\}``:

```math
\\min_{P, Q} \\sum_{(u, i) \\in \\mathcal{S}} \\left( r_{u,i} - \\mathbf{p}_u^{\\mathrm{T}} \\mathbf{q}_i \\right)^2 + \\lambda \\ (\\|\\mathbf{p}_u\\|^2 + \\|\\mathbf{q}_i\\|^2),
```

where ``\\mathbf{p}_u, \\mathbf{q}_i \\in \\mathbb{R}^k`` are respectively a factorized user and item vector, and ``\\lambda`` is a regularization parameter to avoid overfitting. An optimal solution will be found by stochastic gradient descent (SGD). Ultimately, we can predict missing values in ``R`` by just computing ``PQ^{\\mathrm{T}}``, and the prediction directly leads recommendation.
"""
struct MatrixFactorization <: Recommender
    data::DataAccessor
    n_factors::Integer
    P::AbstractMatrix
    Q::AbstractMatrix
    R::AbstractMatrix

    function MatrixFactorization(data::DataAccessor, n_factors::Integer)
        n_users, n_items = size(data.R)
        P = matrix(n_users, n_factors)
        Q = matrix(n_items, n_factors)

        new(data, n_factors, P, Q, matrix(n_users, n_items))
    end
end

"""
    MF(
        data::DataAccessor,
        n_factors::Integer
    )

Alias of `MatrixFactorization`.
"""
const MF = MatrixFactorization

MF(data::DataAccessor) = MF(data, 20)

isdefined(recommender::MatrixFactorization) = isfilled(recommender.R)

function fit!(recommender::MatrixFactorization;
              reg::Float64=1e-3, learning_rate::Float64=1e-3,
              eps::Float64=1e-3, max_iter::Int=100,
              random_init::Bool=false,
              shuffled::Bool=true)
    if random_init
        P = rand(Float64, size(recommender.P))
        Q = rand(Float64, size(recommender.Q))
    else
        # initialize with small constants
        P = ones(size(recommender.P)) * 0.1
        Q = ones(size(recommender.Q)) * 0.1
    end

    nonzero_indices = findall(!iszero, recommender.data.R)
    for it in 1:max_iter
        converged = true

        if shuffled
            shuffle!(nonzero_indices)
        end

        for idx in nonzero_indices
            r = recommender.data.R[idx]

            u, i = idx[1], idx[2]
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
    recommender.R[:] = P * Q'
end

function predict(recommender::MatrixFactorization, user::Integer, item::Integer)
    validate(recommender)
    recommender.R[user, item]
end

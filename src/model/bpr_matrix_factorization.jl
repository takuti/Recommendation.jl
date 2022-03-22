export BPRMatrixFactorization, BPRMF

"""
    BPRMatrixFactorization(
        data::DataAccessor,
        n_factors::Integer
    )

Recommendation based on matrix factorization (MF) with Bayesian personalized ranking (BPR) loss. Number of factors ``k`` is configured by `n_factors`.

- [BPR: Bayesian Personalized Ranking from Implicit Feedback](https://dl.acm.org/doi/10.5555/1795114.1795167)
"""
struct BPRMatrixFactorization <: Recommender
    data::DataAccessor
    n_factors::Integer
    P::AbstractMatrix
    Q::AbstractMatrix

    function BPRMatrixFactorization(data::DataAccessor, n_factors::Integer)
        n_users, n_items = size(data.R)
        P = matrix(n_users, n_factors)
        Q = matrix(n_items, n_factors)

        new(data, n_factors, P, Q)
    end
end

"""
    BPRMF(
        data::DataAccessor,
        n_factors::Integer
    )

Alias of `BPRMatrixFactorization`.
"""
const BPRMF = BPRMatrixFactorization

BPRMF(data::DataAccessor) = BPRMF(data, 20)

isdefined(recommender::BPRMatrixFactorization) = isfilled(recommender.P)

function fit!(recommender::BPRMatrixFactorization;
              reg::Float64=1e-3, learning_rate::Float64=1e-3,
              eps::Float64=1e-3, max_iter::Int=100,
              random_init::Bool=false,
              bootstrap_sampling::Bool=true)
    if random_init
        P = rand(Float64, size(recommender.P))
        Q = rand(Float64, size(recommender.Q))
    else
        # initialize with small constants
        P = ones(size(recommender.P)) * 0.1
        Q = ones(size(recommender.Q)) * 0.1
    end

    samples = get_pairwise_preference_triples(recommender.data.R)

    nnz = count(!iszero, recommender.data.R)
    for _ in 1:max_iter
        loss = 0.0

        batch_size = if bootstrap_sampling
            # optimize by SGD with bootstrap sampling; each step relies on
            # a randomly drawn user-item-item triple, assuming `u` prefers `i` over `j`
            # rather than sequentially iterating all possible samples.
            # the total num of iterations linearly depends on the num of positive (nnz) samples
            nnz
        else
            length(samples)
        end

        for idx in 1:batch_size
            u, i, j = if bootstrap_sampling
                rand(samples)  # random draw
            else
                samples[idx]
            end

            uv, iv, jv = P[u, :], Q[i, :], Q[j, :]

            x_uij = dot(uv, iv) - dot(uv, jv)

            sigmoid = 1 / (1 + exp(-x_uij))
            loss += log(sigmoid)

            P[u, :] = uv .+ learning_rate * ((1 - sigmoid) * (iv .- jv) .+ reg * uv)
            Q[i, :] = iv .+ learning_rate * ((1 - sigmoid) * uv .+ reg * iv)
            Q[j, :] = jv .+ learning_rate * ((1 - sigmoid) * -uv .+ reg * jv)
        end

        if abs(loss / nnz) < eps; break; end;
    end

    recommender.P[:] = P[:]
    recommender.Q[:] = Q[:]
end

function predict(recommender::BPRMatrixFactorization, user::Integer, item::Integer)
    validate(recommender)
    dot(recommender.P[user, :], recommender.Q[item, :])
end

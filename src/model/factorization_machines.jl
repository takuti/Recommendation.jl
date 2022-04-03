export FactorizationMachines

"""
    FactorizationMachines(
        data::DataAccessor,
        n_factors::Integer
    )

Recommendation based on second-order factorization machines (FMs). Number of factors ``k`` is configured by `n_factors`.

Learning FM requires a set of parameters ``\\Theta = \\{w_0, \\mathbf{w}, V\\}`` and a loss function ``\\ell(\\hat{y}(\\mathbf{x} \\mid \\Theta), y)``. Ultimately, the parameters can be optimized by stochastic gradient descent (SGD).
"""
struct FactorizationMachines <: Recommender
    data::DataAccessor
    p::Integer
    n_factors::Integer
    w0::Base.RefValue{Float64} # making mutable
    w::AbstractVector
    V::AbstractMatrix

    function FactorizationMachines(data::DataAccessor, n_factors::Integer)
        n_users, n_items = size(data.R)

        uv = []
        user_vectors = collect(values(data.user_attributes))
        if !isempty(user_vectors)
            uv = user_vectors[1]
        end
        iv = []
        item_vectors = collect(values(data.item_attributes))
        if !isempty(item_vectors)
            iv = item_vectors[1]
        end
        p = n_users + n_items + size(uv, 1) + size(iv, 1)

        w0 = Ref(0.)
        w = vector(p)
        V = matrix(p, n_factors)

        new(data, p, n_factors, w0, w, V)
    end
end

FactorizationMachines(data::DataAccessor) = FactorizationMachines(data, 20)

isdefined(recommender::FactorizationMachines) = isfilled(recommender.V)

function fit!(recommender::FactorizationMachines;
              reg_w0::Float64=1e-3,
              reg_w::Float64=1e-3,
              reg_V::Float64=1e-3,
              learning_rate::Float64=1e-3,
              eps::Float64=1e-3, max_iter::Int=100,
              random_init::Bool=false,
              shuffled::Bool=true)
    if random_init
        w0 = rand()
        w = rand(Float64, size(recommender.w))
        V = rand(Float64, size(recommender.V))
    else
        w0 = 0.
        w = zeros(size(recommender.w))
        # initialize with small constants
        V = ones(size(recommender.V)) * 0.1
    end

    n_users, n_items = size(recommender.data.R)
    nonzero_indices = findall(!iszero, recommender.data.R)

    for it in 1:max_iter
        converged = true

        if shuffled
            shuffle!(nonzero_indices)
        end

        for idx in nonzero_indices
            r = recommender.data.R[idx]

            u, i = idx[1], idx[2]

            u_onehot = zeros(n_users)
            u_onehot[u] = 1.

            i_onehot = zeros(n_items)
            i_onehot[i] = 1.

            x = vcat(u_onehot, i_onehot,
                     get_user_attribute(recommender.data, u),
                     get_item_attribute(recommender.data, i))

            interaction = sum((V' * x).^2 - (V'.^2 * x.^2)) / 2.
            pred = w0 + dot(w, x) + interaction

            err = r - pred
            if abs(err) >= eps; converged = false; end

            w0 = w0 + 2. * learning_rate * (err * 1. - reg_w0 * w0)

            prev_w = copy(w)
            prev_V = copy(V)

            prod = (x' * prev_V)'

            for j in 1:recommender.p
                if x[j] == 0.; continue; end

                w[j] = prev_w[j] + 2. * learning_rate * (err * x[j] - reg_w * prev_w[j])

                g = err * x[j] * (prod - x[j] * prev_V[j, :])
                V[j, :] = prev_V[j, :] + 2. * learning_rate * (g - reg_V * prev_V[j, :])
            end
        end

        if converged; break; end;
    end

    recommender.w0[] = w0
    recommender.w[:] = w[:]
    recommender.V[:] = V[:]
end

function predict(recommender::FactorizationMachines, user::Integer, item::Integer)
    validate(recommender)
    n_users, n_items = size(recommender.data.R)

    u_onehot = zeros(n_users)
    u_onehot[user] = 1.

    i_onehot = zeros(n_items)
    i_onehot[item] = 1.

    x = vcat(u_onehot, i_onehot,
             get_user_attribute(recommender.data, user),
             get_item_attribute(recommender.data, item))

    interaction = sum((recommender.V' * x).^2 - (recommender.V'.^2 * x.^2)) / 2.
    recommender.w0[] + dot(recommender.w, x) + interaction
end

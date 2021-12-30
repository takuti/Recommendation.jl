export FactorizationMachines

"""
    FactorizationMachines(
        data::DataAccessor,
        k::Int
    )
"""
struct FactorizationMachines <: Recommender
    data::DataAccessor
    p::Int
    k::Int
    w0::Base.RefValue{Float64}
    w::AbstractVector
    V::AbstractMatrix

    function FactorizationMachines(data::DataAccessor, k::Int)
        n_user, n_item = size(data.R)

        uv = collect(values(data.user_attributes))[1]
        iv = collect(values(data.item_attributes))[1]
        p = n_user + n_item + size(uv, 1) + size(iv, 1)

        w0 = Ref(0.)
        w = vector(p)
        V = matrix(p, k)

        new(data, p, k, w0, w, V)
    end
end

FactorizationMachines(data::DataAccessor) = FactorizationMachines(data, 20)

isbuilt(recommender::FactorizationMachines) = isfilled(recommender.V)

function build!(recommender::FactorizationMachines;
               reg_w0::Float64=1e-3,
               reg_w::Float64=1e-3,
               reg_V::Float64=1e-3,
               learning_rate::Float64=1e-3,
               eps::Float64=1e-3, max_iter::Int=100)
    n_user, n_item = size(recommender.data.R)

    w0 = 0.
    w = zeros(recommender.p)
    V = rand(recommender.p, recommender.k)

    pairs = vec([(u, i) for u in 1:n_user, i in 1:n_item])
    for it in 1:max_iter
        converged = true

        shuffled_pairs = shuffle(pairs)
        for (u, i) in shuffled_pairs
            r = recommender.data.R[u, i]
            if isnan(r); continue; end

            u_onehot = zeros(n_user)
            u_onehot[u] = 1.

            i_onehot = zeros(n_item)
            i_onehot[i] = 1.

            x = vcat(u_onehot, i_onehot,
                     recommender.data.user_attributes[u],
                     recommender.data.item_attributes[i])

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

    setindex!(recommender.w0, w0)
    recommender.w[:] = w[:]
    recommender.V[:] = V[:]
end

function predict(recommender::FactorizationMachines, u::Int, i::Int)
    check_build_status(recommender)
    n_user, n_item = size(recommender.data.R)

    u_onehot = zeros(n_user)
    u_onehot[u] = 1.

    i_onehot = zeros(n_item)
    i_onehot[i] = 1.

    x = vcat(u_onehot, i_onehot,
             recommender.data.user_attributes[u],
             recommender.data.item_attributes[i])

    interaction = sum((recommender.V' * x).^2 - (recommender.V'.^2 * x.^2)) / 2.
    getindex(recommender.w0) + dot(recommender.w, x) + interaction
end

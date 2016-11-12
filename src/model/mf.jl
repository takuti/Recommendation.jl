export MF

typealias Parameters Dict{Symbol,Any}

immutable MF <: Recommender
    da::DataAccessor
    params::Parameters
    k::Int
end

MF(da::DataAccessor, k::Int) = begin
    n_user, n_item = size(da.R)
    params = Dict(:P => zeros(n_user, k), :Q => zeros(n_item, k))
    MF(da, params, k)
end

function build(recommender::MF;
               reg::Float64=1e-3, learning_rate::Float64=1e-3,
               eps::Float64=1e-3, max_iter::Int=100)
    n_user, n_item = size(recommender.da.R)

    # initialize with small values
    # (random is also possible)
    P = ones(n_user, recommender.k) * 0.1
    Q = ones(n_item, recommender.k) * 0.1

    pairs = vec([(u, i) for u in 1:n_user, i in 1:n_item])
    for it in 1:max_iter
        is_converged = true

        shuffled_pairs = shuffle(pairs)
        for (u, i) in shuffled_pairs
            r = recommender.da.R[u, i]
            if isnan(r); continue; end

            uv, iv = P[u], Q[i]

            err = r - dot(uv, iv)
            if abs(err) >= eps; is_converged = false; end

            grad = -2 * (err * iv - reg * uv)
            P[u] = uv - learning_rate * grad

            grad = -2 * (err * uv - reg * iv)
            Q[i] = iv - learning_rate * grad
        end

        if is_converged; break; end;
    end

    recommender.params[:P] = P
    recommender.params[:Q] = Q
end

function predict(recommender::MF, u::Int, i::Int)
    dot(recommender.params[:P][u, :], recommender.params[:Q][i, :])
end

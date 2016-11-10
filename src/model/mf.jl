export MF

immutable MF <: Recommender
    m::AbstractMatrix
    m_approx::AbstractMatrix
    P::AbstractMatrix
    Q::AbstractMatrix
    k::Int
end

MF(m::AbstractMatrix, k::Int;
   reg::Float64=1e-3, learning_rate::Float64=1e-3,
   eps::Float64=1e-3, max_iter::Int=100) = begin

    n_user, n_item = size(m)

    # initialize with small values
    # (random is also possible)
    P = ones(n_user, k) * 0.1
    Q = ones(n_item, k) * 0.1

    pairs = vec([(u, i) for u in 1:n_user, i in 1:n_item])
    for it in 1:max_iter
        is_converged = true

        shuffled_pairs = shuffle(pairs)
        for (u, i) in shuffled_pairs
            if isnan(m[u, i]); continue; end

            uv, iv = P[u], Q[i]

            err = m[u, i] - dot(uv, iv)
            if abs(err) >= eps; is_converged = false; end

            grad = -2 * (err * iv - reg * uv)
            P[u] = uv - learning_rate * grad

            grad = -2 * (err * uv - reg * iv)
            Q[i] = iv - learning_rate * grad
        end

        if is_converged; break; end;
    end

    m_approx = P * Q'
    MF(m, m_approx, P, Q, k)
end

function predict(recommender::MF, u::Int, i::Int)
    recommender.m_approx[u, i]
end

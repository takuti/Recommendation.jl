export UserKNN

"""
    UserKNN(
        data::DataAccessor,
        n_neighbors::Integer,
        normalize::Bool=false
    )

[User-based CF using the Pearson correlation](https://dl.acm.org/citation.cfm?id=312682). `n_neighbors` represents number of neighbors ``k``, and `normalize` specifies if weighted sum of neighbors' rating is normalized.

The technique gives a weight to a user-user pair by the following equation:

```math
s_{u, v} = \\frac{ \\sum_{i \\in \\mathcal{I}^+_{u \\cap v}}  (r_{u, i} - \\overline{r}_u) (r_{v, i} - \\overline{r}_v)}
{ \\sqrt{\\sum_{i \\in \\mathcal{I}^+_{u \\cap v}} (r_{u,i} - \\overline{r}_u)^2} \\sqrt{\\sum_{i \\in \\mathcal{I}^+_{u \\cap v}} (r_{v, i} - \\overline{r}_v)^2} },
```

where ``\\mathcal{I}^+_{u \\cap v}`` is a set of items which were observed by both user ``u`` and ``v``, and ``r_{u, i}`` indicates a ``(u, i)`` element in ``R``. Plus, ``\\overline{r}_u`` and ``\\overline{r}_v`` are respectively mean values of ``r_{u, i}`` and ``r_{v, i}`` over ``i \\in \\mathcal{I}^+_{u \\cap v}``. By using the weights, user-based CF selects the top-``k`` highest-weighted users (i.e., nearest neighbors) of a target user ``u``, and predicts missing ``r_{u, i}`` for ``i \\in \\mathcal{I}^-_u`` as:

```math
r_{u, i} = \\overline{r}_u + \\frac{\\sum^k_{t=1} \\left(r_{\\sigma(t), i} - \\overline{r}_{\\sigma(t)}\\right) \\cdot s_{u,\\sigma(t)} }{ \\sum^k_{t=1} s_{u,\\sigma(t)} },
```

where ``\\sigma(t)`` denotes the ``t``-th nearest-neighborhood user. Ultimately, sorting items ``\\mathcal{I}^-_u`` by the predicted values enables us to make recommendation to a user ``u``.

It should be noted that user-based CF is highly inefficient because gradually increasing massive users and their dynamic tastes require us to frequently recompute the similarities for every pairs of users.
"""
struct UserKNN <: Recommender
    data::DataAccessor
    n_neighbors::Integer
    sim::AbstractMatrix
    normalize::Bool

    function UserKNN(data::DataAccessor, n_neighbors::Integer, normalize::Bool)
        n_users = size(data.R, 1)
        n_neighbors = min(n_users - 1, n_neighbors)  # max #neighbors is (#users - 1), excluding a target user him/herself
        new(data, n_neighbors, matrix(n_users, n_users), normalize)
    end
end

UserKNN(data::DataAccessor, n_neighbors::Integer) = UserKNN(data, n_neighbors, false)
UserKNN(data::DataAccessor) = UserKNN(data, 20, false)

isdefined(recommender::UserKNN) = isfilled(recommender.sim)

function fit!(recommender::UserKNN)
    # Pearson correlation
    nonzero_flags = (!iszero).(recommender.data.R)
    n_users = size(recommender.data.R, 1)
    for ui in 1:n_users
        for uj in ui:n_users
            # pairwise correlation
            # (zeros, which might originally be unknown values, are ignored)
            co_occurred_indices = nonzero_flags[ui, :] .& nonzero_flags[uj, :]

            vi = recommender.data.R[ui, co_occurred_indices] .- mean(recommender.data.R[ui, co_occurred_indices])
            vj = recommender.data.R[uj, co_occurred_indices] .- mean(recommender.data.R[uj, co_occurred_indices])

            denom = length(vi) == 0 ? 0 : sqrt(dot(vi, vi) * dot(vj, vj)) # zero similarity if there is no co-occurred index
            similarity = iszero(denom) ? 0 : (dot(vi, vj) / denom)

            recommender.sim[ui, uj] = similarity
            if (ui != uj); recommender.sim[uj, ui] = similarity; end # symmetric
        end
    end
end

function predict(recommender::UserKNN, user::Integer, item::Integer)
    validate(recommender)

    user_similarity_pairs = collect(enumerate(recommender.sim[user, :]))

    # closest neighbor is always target user him/herself, so omit him/her
    neighbors = sort(user_similarity_pairs, by=tuple->last(tuple), rev=true)[2:(recommender.n_neighbors + 1)]

    numer = denom = 0
    for (u_near, w) in neighbors
        v_near = recommender.data.R[u_near, :]

        r = v_near[item]
        if iszero(r); continue; end

        if recommender.normalize
            jj = (!iszero).(v_near)
            r -= mean(v_near[jj])
        end

        numer += r * w
        denom += w
    end
    pred = iszero(denom) ? 0 : numer / denom

    if recommender.normalize
        ii = (!iszero).(recommender.data.R[user, :])
        pred += mean(recommender.data.R[user, ii])
    end

    pred
end

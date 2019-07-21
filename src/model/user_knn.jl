export UserKNN

"""
    UserKNN(
        data::DataAccessor,
        k::Int,
        normalize::Bool=false
    )

[User-based CF using the Pearson correlation](https://dl.acm.org/citation.cfm?id=312682). `k` represents number of neighbors, and `normalize` specifies if weighted sum of neighbors' rating is normalized.

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
    k::Int
    sim::AbstractMatrix
    normalize::Bool

    function UserKNN(data::DataAccessor, k::Int, normalize::Bool)
        n_user = size(data.R, 1)
        new(data, k, matrix(n_user, n_user), normalize)
    end
end

UserKNN(data::DataAccessor, k::Int) = UserKNN(data, k, false)
UserKNN(data::DataAccessor) = UserKNN(data, 20, false)

isbuilt(recommender::UserKNN) = isfilled(recommender.sim)

function build!(recommender::UserKNN)
    # Pearson correlation

    R = copy(recommender.data.R)

    n_row = size(R, 1)

    for ri in 1:n_row
        for rj in ri:n_row
            # pairwise correlation (i.e., ignore NaNs)
            ij = broadcast(!isnan, R[ri, :]) .& broadcast(!isnan, R[rj, :])

            vi = R[ri, :] .- mean(R[ri, ij])
            vj = R[rj, :] .- mean(R[rj, ij])

            numer = dot(vi[ij], vj[ij])
            denom = sqrt(dot(vi[ij], vi[ij]) * dot(vj[ij], vj[ij]))

            c = numer / denom
            recommender.sim[ri, rj] = c
            if (ri != rj); recommender.sim[rj, ri] = c; end # symmetric
        end
    end
end

function predict(recommender::UserKNN, u::Int, i::Int)
    check_build_status(recommender)

    numer = denom = 0

    pairs = collect(zip(1:size(recommender.data.R)[1], recommender.sim[u, :]))
    # closest neighbor is always target user him/herself, so omit him/her
    ordered_pairs = sort(pairs, by=tuple->last(tuple), rev=true)[2:(recommender.k + 1)]

    for (u_near, w) in ordered_pairs
        v_near = recommender.data.R[u_near, :]

        r = v_near[i]
        if isnan(r); continue; end

        r_ = 0
        if recommender.normalize
            jj = broadcast(!isnan, v_near)
            r_ = mean(v_near[jj])
        end

        numer += (r - r_) * w
        denom += w
    end

    pred = (denom == 0) ? 0 : numer / denom
    if recommender.normalize
        ii = broadcast(!isnan, recommender.data.R[u, :])
        pred += mean(recommender.data.R[u, ii])
    end
    pred
end

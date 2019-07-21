export ItemKNN

"""
    ItemKNN(
        data::DataAccessor,
        k::Int
    )

[Item-based CF](https://dl.acm.org/citation.cfm?id=963776) that provides a way to model item-item concepts by utilizing the similarities of items in the CF paradigm. `k` represents number of neighbors.

Item properties are relatively stable compared to the users' tastes, and the number of items is generally smaller than the number of users. Hence, while user-based CF successfully captures the similarities of users' complex tastes, modeling item-item concepts could be much more promising in terms of both scalability and overall accuracy.

Item-based CF defines a similarity between an item ``i`` and ``j`` as:

```math
s_{i,j} = \\frac{ \\sum_{u \\in \\mathcal{U}_{i \\cap j}}  (r_{u, i} - \\overline{r}_i) (r_{u, j} - \\overline{r}_j)}
{ \\sqrt{\\sum_{u \\in \\mathcal{U}_{i \\cap j}} (r_{u,i} - \\overline{r}_i)^2} \\sqrt{\\sum_{u \\in \\mathcal{U}_{i \\cap j}} (r_{u, j} - \\overline{r}_j)^2} },
```

where ``\\mathcal{U}_{i \\cap j}`` is a set of users that both of ``r_{u,i}`` and ``r_{u, j}`` are not missing, and ``\\overline{r}_i, \\overline{r}_j`` are mean values of ``i``-th and ``j``-th column in ``R``. Similarly to the user-based algorithm, for the ``t``-th nearest-neighborhood item ``\\tau(t)``, prediction can be done by top-``k`` weighted sum of target user's feedbacks:

```math
r_{u,i} = \\frac{\\sum^k_{t=1} s_{i,\\tau(t)} \\cdot r_{u,\\tau(t)} }{ \\sum^k_{t=1} s_{i,\\tau(t)} }.
```

In case that the number of items is smaller than users, item-based CF could be a more reasonable choice than the user-based approach.
"""
struct ItemKNN <: Recommender
    data::DataAccessor
    k::Int
    sim::AbstractMatrix

    function ItemKNN(data::DataAccessor, k::Int)
        n_item = size(data.R, 2)
        new(data, k, matrix(n_item, n_item))
    end
end

ItemKNN(data::DataAccessor) = ItemKNN(data, 5)

isbuilt(recommender::ItemKNN) = isfilled(recommender.sim)

function build!(recommender::ItemKNN; adjusted_cosine::Bool=false)
    # cosine similarity

    R = copy(recommender.data.R)
    n_row, n_col = size(R)

    if adjusted_cosine
        # subtract mean
        for ri in 1:n_row
            indices = broadcast(!isnan, R[ri, :])
            vmean = mean(R[ri, indices])
            R[ri, indices] .-= vmean
        end
    end

    # unlike pearson correlation, matrix can be filled by zeros for cosine similarity
    R[isnan.(R)] .= 0

    # compute L2 nrom of each column
    norms = sqrt.(sum(R.^2, dims=1))

    for ci in 1:n_col
        for cj in ci:n_col
            numer = dot(R[:, ci], R[:, cj])
            denom = norms[ci] * norms[cj]
            s = numer / denom

            recommender.sim[ci, cj] = s
            if (ci != cj); recommender.sim[cj, ci] = s; end
        end
    end

    # NaN similarities are converted into zeros
    recommender.sim[isnan.(recommender.sim)] .= 0
end

function predict(recommender::ItemKNN, u::Int, i::Int)
    check_build_status(recommender)

    numer = denom = 0

    # negative similarities are filtered
    pairs = collect(zip(1:size(recommender.data.R)[2], max.(recommender.sim[i, :], 0)))
    ordered_pairs = sort(pairs, by=tuple->last(tuple), rev=true)[1:recommender.k]

    for (j, s) in ordered_pairs
        r = recommender.data.R[u, j]
        if isnan(r); continue; end

        numer += s * r
        denom += s
    end

    (denom == 0) ? 0 : numer / denom
end

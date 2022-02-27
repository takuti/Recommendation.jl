export ItemKNN

"""
    ItemKNN(
        data::DataAccessor,
        n_neighbors::Integer
    )

[Item-based CF](https://dl.acm.org/citation.cfm?id=963776) that provides a way to model item-item concepts by utilizing the similarities of items in the CF paradigm. `n_neighbors` represents number of neighbors ``k``.

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
    n_neighbors::Integer
    sim::AbstractMatrix

    function ItemKNN(data::DataAccessor, n_neighbors::Integer)
        n_items = size(data.R, 2)
        n_neighbors = min(n_items, n_neighbors)
        new(data, n_neighbors, matrix(n_items, n_items))
    end
end

ItemKNN(data::DataAccessor) = ItemKNN(data, 5)

isdefined(recommender::ItemKNN) = isfilled(recommender.sim)

function fit!(recommender::ItemKNN; adjusted_cosine::Bool=false)
    # cosine similarity
    if adjusted_cosine
        # subtract mean (of nonzero elements) from the matrix, and keep zero elements as-is
        nonzero_flags = (!iszero).(recommender.data.R)
        R = (recommender.data.R .- (sum(recommender.data.R, dims=2) ./ sum(nonzero_flags, dims=2))) .* nonzero_flags
    else
        R = copy(recommender.data.R)
    end

    # compute L2 nrom of each column
    norms = sqrt.(sum(R.^2, dims=1))

    n_items = size(R, 2)
    for ii in 1:n_items
        for ij in ii:n_items
            denom = norms[ii] * norms[ij]
            similarity = iszero(denom) ? 0 : (dot(R[:, ii], R[:, ij]) / denom)
            recommender.sim[ii, ij] = similarity
            if (ii != ij); recommender.sim[ij, ii] = similarity; end
        end
    end
end

function predict(recommender::ItemKNN, u::Integer, i::Integer)
    validate(recommender)

    # filter out negative similarities
    item_similarity_pairs = collect(enumerate(max.(recommender.sim[i, :], 0)))
    neighbors = sort(item_similarity_pairs, by=tuple->last(tuple), rev=true)[1:recommender.n_neighbors]

    numer = denom = 0
    for (i_near, s) in neighbors
        r = recommender.data.R[u, i_near]
        if iszero(r); continue; end

        numer += s * r
        denom += s
    end
    iszero(denom) ? 0 : numer / denom
end

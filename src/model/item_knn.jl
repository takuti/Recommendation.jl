export ItemKNN

immutable ItemKNN <: Recommender
    da::DataAccessor
    sim::AbstractMatrix
    k::Int
end

ItemKNN(da::DataAccessor, k::Int) = begin
    n_item = size(da.R, 2)
    ItemKNN(da, zeros(n_item, n_item), k)
end

function build(recommender::ItemKNN; is_adjusted_cosine::Bool=false)
    # cosine similarity

    R = copy(recommender.da.R)
    n_row, n_col = size(R)

    if is_adjusted_cosine
        # subtract mean
        for ri in 1:n_row
            indices = !isnan(R[ri, :])
            vmean = mean(R[ri, indices])
            R[ri, indices] -= vmean
        end
    end

    # unlike pearson correlation, matrix can be filled by zeros for cosine similarity
    R[isnan(R)] = 0

    # compute L2 nrom of each column
    norms = sqrt(sum(R.^2, 1))

    for ci in 1:n_col
        for cj in ci:n_col
            numer = dot(R[:, ci], R[:, cj])
            denom = norms[ci] * norms[cj]
            s = numer / denom

            recommender.sim[ci, cj] = s
            if (ci != cj); recommender.sim[cj, ci] = s; end
        end
    end
end

function predict(recommender::ItemKNN, u::Int, i::Int)
    numer = denom = 0

    # negative similarities are filtered
    pairs = collect(zip(1:size(recommender.da.R)[2], max(recommender.sim[i, :], 0)))
    ordered_pairs = sort(pairs, by=tuple->last(tuple), rev=true)[1:recommender.k]

    for (j, s) in ordered_pairs
        r = recommender.da.R[u, j]
        if isnan(r); continue; end

        numer += s * r
        denom += s
    end

    (denom == 0) ? 0 : numer / denom
end

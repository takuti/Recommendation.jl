export ItemKNN

struct ItemKNN <: Recommender
    da::DataAccessor
    hyperparams::Parameters
    sim::AbstractMatrix
    states::States
end

ItemKNN(da::DataAccessor,
        hyperparams::Parameters=Parameters(:k => 5)) = begin
    n_item = size(da.R, 2)
    ItemKNN(da, hyperparams, zeros(n_item, n_item), States(:is_built => false))
end

function build(rec::ItemKNN; is_adjusted_cosine::Bool=false)
    # cosine similarity

    R = copy(rec.da.R)
    n_row, n_col = size(R)

    if is_adjusted_cosine
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

            rec.sim[ci, cj] = s
            if (ci != cj); rec.sim[cj, ci] = s; end
        end
    end

    # NaN similarities are converted into zeros
    rec.sim[isnan.(rec.sim)] .= 0

    rec.states[:is_built] = true
end

function predict(rec::ItemKNN, u::Int, i::Int)
    check_build_status(rec)

    numer = denom = 0

    # negative similarities are filtered
    pairs = collect(zip(1:size(rec.da.R)[2], max.(rec.sim[i, :], 0)))
    ordered_pairs = sort(pairs, by=tuple->last(tuple), rev=true)[1:rec.hyperparams[:k]]

    for (j, s) in ordered_pairs
        r = rec.da.R[u, j]
        if isnan(r); continue; end

        numer += s * r
        denom += s
    end

    (denom == 0) ? 0 : numer / denom
end

export ItemKNN

immutable ItemKNN <: Recommender
    m::AbstractMatrix
    sim::AbstractMatrix
    k::Int
    is_normalized::Bool
end

ItemKNN(m::AbstractMatrix, k::Int; is_normalized::Bool=false) = begin
    n_user, n_item = size(m)
    sim = zeros(n_item, n_item)

    m_ = copy(m)
    if is_normalized
        # subtract mean
        for u in 1:n_user
            indices = !isnan(m_[u, :])
            vmean = mean(m_[u, indices])
            m_[u, indices] -= vmean
        end
    end

    # unlike pearson correlation, matrix can be filled by zeros for cosine similarity
    m_[isnan(m_)] = 0

    # compute L2 nrom of each column
    norms = sqrt(sum(m_.^2, 1))

    for ii in 1:n_item
        for ij in ii:n_item
            numer = dot(m_[:, ii], m_[:, ij])
            denom = norms[ii] * norms[ij]
            s = numer / denom

            sim[ii, ij] = s
            if (ii != ij); sim[ij, ii] = s; end
        end
    end

    ItemKNN(m, sim, k, is_normalized)
end

function predict(recommender::ItemKNN, u::Int, i::Int)
    numer = denom = 0

    # negative similarities are filtered
    pairs = collect(zip(1:size(recommender.m)[2], max(recommender.sim[i, :], 0)))
    ordered_pairs = sort(pairs, by=tuple->last(tuple), rev=true)[1:recommender.k]

    for (j, s) in ordered_pairs
        r = recommender.m[u, j]
        if isnan(r); continue; end

        numer += s * r
        denom += s
    end

    (denom == 0) ? 0 : numer / denom
end

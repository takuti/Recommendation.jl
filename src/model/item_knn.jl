export ItemKNN

immutable ItemKNN <: Recommender
    m::AbstractMatrix
    sim::AbstractMatrix
    k::Int
    is_normalized::Bool
end

ItemKNN(m::AbstractMatrix, k::Int; is_normalized::Bool=false) = begin
    sim = MatrixUtils.cosine_similarity(m, 2, is_normalized=is_normalized)
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

export ItemKNN

immutable ItemKNN <: Recommender
    m::AbstractMatrix
    sim::AbstractMatrix
    k::Int
end

ItemKNN(m::AbstractMatrix, k::Int;
        similarity="pearson", is_normalized_cosine::Bool=false) = begin

    if similarity == "pearson"
        sim = MatrixUtils.pearson_correlation(m, 2)
    elseif similarity == "cosine"
        sim = MatrixUtils.cosine_similarity(m, 2, is_normalized_cosine)
    end

    ItemKNN(m, sim, k)
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

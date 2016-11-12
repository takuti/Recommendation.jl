export ItemKNN

immutable ItemKNN <: Recommender
    da::DataAccessor
    sim::AbstractMatrix
    k::Int
end

ItemKNN(da::DataAccessor, k::Int;
        similarity="pearson", is_adjusted_cosine::Bool=false) = begin

    if similarity == "pearson"
        sim = MatrixUtils.pearson_correlation(da.R, 2)
    elseif similarity == "cosine"
        sim = MatrixUtils.cosine_similarity(da.R, 2, is_adjusted_cosine)
    end

    ItemKNN(da, sim, k)
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

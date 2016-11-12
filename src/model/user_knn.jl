export UserKNN

immutable UserKNN <: Recommender
    da::DataAccessor
    sim::AbstractMatrix
    k::Int
    is_normalized::Bool
end

UserKNN(da::DataAccessor, k::Int;
        similarity="pearson", is_adjusted_cosine::Bool=false,
        is_normalized::Bool=false) = begin

    if similarity == "pearson"
        sim = MatrixUtils.pearson_correlation(da.R, 1)
    elseif similarity == "cosine"
        sim = MatrixUtils.cosine_similarity(da.R, 1, is_adjusted_cosine)
    end

    UserKNN(da, sim, k, is_normalized)
end

function predict(recommender::UserKNN, u::Int, i::Int)
    numer = denom = 0

    pairs = collect(zip(1:size(recommender.da.R)[1], recommender.sim[u, :]))
    # closest neighbor is always target user him/herself, so omit him/her
    ordered_pairs = sort(pairs, by=tuple->last(tuple), rev=true)[2:(recommender.k + 1)]

    for (u_near, w) in ordered_pairs
        v_near = recommender.da.R[u_near, :]

        r = v_near[i]
        if isnan(r); continue; end

        r_ = 0
        if recommender.is_normalized
            jj = !isnan(v_near)
            r_ = mean(v_near[jj])
        end

        numer += (r - r_) * w
        denom += w
    end

    pred = (denom == 0) ? 0 : numer / denom
    if recommender.is_normalized
        ii = !isnan(recommender.da.R[u, :])
        pred += mean(recommender.da.R[u, ii])
    end
    pred
end

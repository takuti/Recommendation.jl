export UserKNN

immutable UserKNN <: Recommender
    m::AbstractMatrix
    sim::AbstractMatrix
    k::Int
    is_normalized_pred::Bool
end

UserKNN(m::AbstractMatrix, k::Int;
        similarity="pearson", is_normalized_cosine::Bool=false,
        is_normalized_pred::Bool=false) = begin

    if similarity == "pearson"
        sim = MatrixUtils.pearson_correlation(m, 1)
    elseif similarity == "cosine"
        sim = MatrixUtils.cosine_similarity(m, 1, is_normalized_cosine)
    end

    UserKNN(m, sim, k, is_normalized_pred)
end

function predict(recommender::UserKNN, u::Int, i::Int)
    numer = denom = 0

    pairs = collect(zip(1:size(recommender.m)[1], recommender.sim[u, :]))
    # closest neighbor is always target user him/herself, so omit him/her
    ordered_pairs = sort(pairs, by=tuple->last(tuple), rev=true)[2:(recommender.k + 1)]

    for (u_near, w) in ordered_pairs
        v_near = recommender.m[u_near, :]

        r = v_near[i]
        if isnan(r); continue; end

        r_ = 0
        if recommender.is_normalized_pred
            jj = !isnan(v_near)
            r_ = mean(v_near[jj])
        end

        numer += (r - r_) * w
        denom += w
    end

    pred = (denom == 0) ? 0 : numer / denom
    if recommender.is_normalized_pred
        ii = !isnan(recommender.m[u, :])
        pred += mean(recommender.m[u, ii])
    end
    pred
end

export UserKNN

immutable UserKNN <: Recommender
    m::AbstractMatrix
    corr::AbstractMatrix
    k::Int
    is_normalized::Bool
end

UserKNN(m::AbstractMatrix, k::Int; is_normalized::Bool=false) = begin
    corr = MatrixUtils.pearson_correlation(m, 1)
    UserKNN(m, corr, k, is_normalized)
end

function predict(recommender::UserKNN, u::Int, i::Int)
    numer = denom = 0

    pairs = collect(zip(1:size(recommender.m)[1], recommender.corr[u, :]))
    # closest neighbor is always target user him/herself, so omit him/her
    ordered_pairs = sort(pairs, by=tuple->last(tuple), rev=true)[2:(recommender.k + 1)]

    for (u_near, w) in ordered_pairs
        v_near = recommender.m[u_near, :]

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
        ii = !isnan(recommender.m[u, :])
        pred += mean(recommender.m[u, ii])
    end
    pred
end

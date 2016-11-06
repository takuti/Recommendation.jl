export UserKNN

immutable UserKNN <: Recommender
    m::AbstractMatrix
    corr::AbstractMatrix
    k::Int
    is_normalized::Bool
end

UserKNN(m::AbstractMatrix, k::Int; is_normalized::Bool=false) = begin
    n_user = size(m)[1]
    corr = zeros(n_user, n_user)

    for ui in 1:n_user
        for uj in ui:n_user
            # pairwise correlation (i.e., ignore NaNs)
            ij = !isnan(m[ui, :]) & !isnan(m[uj, :])

            vi = m[ui, :] - mean(m[ui, ij])
            vj = m[uj, :] - mean(m[uj, ij])

            numer = dot(vi[ij], vj[ij])
            denom = sqrt(dot(vi[ij], vi[ij]) * dot(vj[ij], vj[ij]))

            c = numer / denom
            corr[ui, uj] = c
            if (ui != uj); corr[uj, ui] = c; end
        end
    end

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

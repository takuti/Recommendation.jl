export UserKNN

immutable UserKNN <: Recommender
    da::DataAccessor
    hyperparams::Parameters
    sim::AbstractMatrix
    is_normalized::Bool
    states::States
end

UserKNN(da::DataAccessor,
        hyperparams::Parameters=Parameters(:k => 5);
        is_normalized::Bool=false) = begin
    n_user = size(da.R, 1)
    UserKNN(da, hyperparams, zeros(n_user, n_user), is_normalized,
            States(:is_built => false))
end

function build(rec::UserKNN)
    # Pearson correlation

    R = copy(rec.da.R)

    n_row = size(R, 1)

    for ri in 1:n_row
        for rj in ri:n_row
            # pairwise correlation (i.e., ignore NaNs)
            ij = !isnan.(R[ri, :]) & !isnan.(R[rj, :])

            vi = R[ri, :] - mean(R[ri, ij])
            vj = R[rj, :] - mean(R[rj, ij])

            numer = dot(vi[ij], vj[ij])
            denom = sqrt(dot(vi[ij], vi[ij]) * dot(vj[ij], vj[ij]))

            c = numer / denom
            rec.sim[ri, rj] = c
            if (ri != rj); rec.sim[rj, ri] = c; end # symmetric
        end
    end

    rec.states[:is_built] = true
end

function predict(rec::UserKNN, u::Int, i::Int)
    check_build_status(rec)

    numer = denom = 0

    pairs = collect(zip(1:size(rec.da.R)[1], rec.sim[u, :]))
    # closest neighbor is always target user him/herself, so omit him/her
    ordered_pairs = sort(pairs, by=tuple->last(tuple), rev=true)[2:(rec.hyperparams[:k] + 1)]

    for (u_near, w) in ordered_pairs
        v_near = rec.da.R[u_near, :]

        r = v_near[i]
        if isnan(r); continue; end

        r_ = 0
        if rec.is_normalized
            jj = !isnan.(v_near)
            r_ = mean(v_near[jj])
        end

        numer += (r - r_) * w
        denom += w
    end

    pred = (denom == 0) ? 0 : numer / denom
    if rec.is_normalized
        ii = !isnan.(rec.da.R[u, :])
        pred += mean(rec.da.R[u, ii])
    end
    pred
end

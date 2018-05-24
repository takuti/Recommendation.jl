export SVD

struct SVD <: Recommender
    da::DataAccessor
    hyperparams::Parameters
    params::Parameters
    states::States
end

SVD(da::DataAccessor,
    hyperparams::Parameters=Parameters(:k => 20)) = begin
    n_user, n_item = size(da.R)
    params = Parameters(:U => zeros(n_user, hyperparams[:k]),
                        :S => zeros(hyperparams[:k]),
                        :V => zeros(n_item, hyperparams[:k]))
    SVD(da, hyperparams, params, States(:is_built => false))
end

function build(rec::SVD)
    # NaNs are filled by zeros for now
    R = copy(rec.da.R)
    R[isnan.(R)] .= 0

    res = svdfact(R)
    rec.params[:U] = res.U[:, 1:rec.hyperparams[:k]]
    rec.params[:S] = res.S[1:rec.hyperparams[:k]]
    rec.params[:Vt] = res.Vt[1:rec.hyperparams[:k], :]

    rec.states[:is_built] = true
end

function predict(rec::SVD, u::Int, i::Int)
    check_build_status(rec)
    dot(rec.params[:U][u, :] .* rec.params[:S], rec.params[:Vt][:, i])
end

export SVD

immutable SVD <: Recommender
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
    R[isnan.(R)] = 0

    res = svds(R, nsv=rec.hyperparams[:k])[1]
    rec.params[:U] = res.U
    rec.params[:S] = res.S
    rec.params[:V] = res.Vt

    rec.states[:is_built] = true
end

function predict(rec::SVD, u::Int, i::Int)
    check_build_status(rec)
    dot(rec.params[:U][u, :] .* rec.params[:S], rec.params[:V][i, :])
end

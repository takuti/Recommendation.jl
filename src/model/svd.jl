export SVD

typealias Parameters Dict{Symbol,Any}

immutable SVD <: Recommender
    da::DataAccessor
    params::Parameters
    k::Int
end

SVD(da::DataAccessor, k::Int) = begin
    n_user, n_item = size(da.R)
    params = Dict(:U => zeros(n_user, k),
                  :S => zeros(k),
                  :V => zeros(n_item, k))
    SVD(da, params, k)
end

function build(recommender::SVD)
    # NaNs are filled by zeros for now
    R = copy(recommender.da.R)
    R[isnan(R)] = 0

    res = svds(R, nsv=recommender.k)[1]
    recommender.params[:U] = res.U
    recommender.params[:S] = res.S
    recommender.params[:V] = res.Vt
end

function predict(recommender::SVD, u::Int, i::Int)
    dot(recommender.params[:U][u, :] .* recommender.params[:S], recommender.params[:V][i, :])
end

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

function build(rec::SVD)
    # NaNs are filled by zeros for now
    R = copy(rec.da.R)
    R[isnan(R)] = 0

    res = svds(R, nsv=rec.k)[1]
    rec.params[:U] = res.U
    rec.params[:S] = res.S
    rec.params[:V] = res.Vt
end

function predict(rec::SVD, u::Int, i::Int)
    dot(rec.params[:U][u, :] .* rec.params[:S], rec.params[:V][i, :])
end

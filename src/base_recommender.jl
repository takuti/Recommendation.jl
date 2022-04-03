export Recommender
export isdefined, validate, fit!, recommend, predict

abstract type Recommender end

function validate(recommender::Recommender)
    if !isdefined(recommender)
        error("Recommender $(typeof(recommender)) is not built before making recommendation")
    end
end

function validate(recommender::Recommender, data::DataAccessor)
    validate(recommender)

    n_rec_users, n_rec_items = size(recommender.data.R)
    n_data_users, n_data_items = size(data.R)

    if n_rec_users != n_data_users
        error("number of users is mismatched: (recommender, target) = ($(n_rec_users), $(n_data_users)")
    elseif n_rec_items != n_data_items
        error("number of items is mismatched: (recommender, target) = ($(n_rec_items), $(n_data_items)")
    end
end

isdefined(recommender::Recommender) = true

function fit!(recommender::Recommender; kwargs...)
    error("fit! is not implemented for recommender type $(typeof(recommender))")
end

function recommend(recommender::Recommender, user::Integer, topk::Integer, candidates::AbstractVector{T}) where {T<:Integer}
    d = Dict{T,AbstractFloat}()
    for candidate in candidates
        score = predict(recommender, user, candidate)
        if isnan(score); continue; end
        d[candidate] = score
    end
    ranked_recs = sort(collect(d), lt=((k1,v1), (k2,v2)) -> v1>v2 || ((v1==v2) && k1<k2))
    ranked_recs[1:min(length(ranked_recs), topk)]
end

function predict(recommender::Recommender, user::Integer, item::Integer)
    error("predict is not implemented for recommender type $(typeof(recommender))")
end

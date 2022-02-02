export Recommender
export isdefined, validate, fit!, recommend, predict, ranking

abstract type Recommender end

function validate(recommender::Recommender)
    if !isdefined(recommender)
        error("Recommender $(typeof(recommender)) is not built before making recommendation")
    end
end

function validate(recommender::Recommender, data::DataAccessor)
    validate(recommender)

    n_rec_user, n_rec_item = size(recommender.data.R)
    n_data_user, n_data_item = size(data.R)

    if n_rec_user != n_data_user
        error("number of users is mismatched: (recommender, target) = ($(n_rec_user), $(n_data_user)")
    elseif n_rec_item != n_data_item
        error("number of items is mismatched: (recommender, target) = ($(n_rec_item), $(n_data_item)")
    end
end

isdefined(recommender::Recommender) = true

function fit!(recommender::Recommender; kwargs...)
    error("fit! is not implemented for recommender type $(typeof(recommender))")
end

function recommend(recommender::Recommender, u::Integer, k::Integer, candidates::Array{T}) where {T<:Integer}
    d = Dict{T,AbstractFloat}()
    for candidate in candidates
        score = ranking(recommender, u, candidate)
        if isnan(score); continue; end
        d[candidate] = score
    end
    ranked_recs = sort(collect(d), lt=((k1,v1), (k2,v2)) -> v1>v2 || ((v1==v2) && k1<k2))
    ranked_recs[1:min(length(ranked_recs), k)]
end

function predict(recommender::Recommender, u::Integer, i::Integer)
    error("predict is not implemented for recommender type $(typeof(recommender))")
end

# Return a ranking score of item i for user u
function ranking(recommender::Recommender, u::Integer, i::Integer)
    validate(recommender)
    predict(recommender, u, i)
end

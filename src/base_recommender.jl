export Recommender
export isbuilt, check_build_status, build!, recommend, predict, ranking

abstract type Recommender end

function check_build_status(recommender::Recommender)
    if !isbuilt(recommender)
        error("Recommender $(typeof(recommender)) is not built before making recommendation")
    end
end

isbuilt(recommender::Recommender) = true

function build!(recommender::Recommender; kwargs...)
    error("build! is not implemented for recommender type $(typeof(recommender))")
end

function recommend(recommender::Recommender, u::Int, k::Int, candidates::Array{Int})
    d = Dict{Int,Float64}()
    for candidate in candidates
        score = ranking(recommender, u, candidate)
        if isnan(score); continue; end
        d[candidate] = score
    end
    ranked_recs = sort(collect(d), lt=((k1,v1), (k2,v2)) -> v1>v2 || ((v1==v2) && k1<k2))
    ranked_recs[1:min(length(ranked_recs), k)]
end

function predict(recommender::Recommender, u::Int, i::Int)
    error("predict is not implemented for recommender type $(typeof(recommender))")
end

# Return a ranking score of item i for user u
function ranking(recommender::Recommender, u::Int, i::Int)
    check_build_status(recommender)
    predict(recommender, u, i)
end

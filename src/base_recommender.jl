export Recommender, ContentRecommender
export build, execute, predict, ranking

abstract Recommender

function build(rec::Recommender; kwargs...)
    error("build is not implemented for recommender type $(typeof(rec))")
end

function execute(rec::Recommender, u::Int, k::Int, candidates::Array{Int})
    d = Dict{Int,Float64}()
    for candidate in candidates
        score = ranking(rec, u, candidate)
        d[candidate] = score
    end
    sort(collect(d), by=tuple->last(tuple), rev=true)[1:k]
end

function predict(rec::Recommender, u::Int, i::Int)
    error("predict is not implemented for recommender type $(typeof(rec))")
end

# Return a ranking score of item i for user u
function ranking(rec::Recommender, u::Int, i::Int)
    predict(rec, u, i)
end

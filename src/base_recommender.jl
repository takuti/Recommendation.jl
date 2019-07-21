export Recommender
export check_build_status, build!, recommend, predict, ranking

abstract type Recommender end

function check_build_status(rec::Recommender)
    if !haskey(rec.states, :built) || !rec.states[:built]
        error("Recommender $(typeof(rec)) is not built before making recommendation")
    end
end

function build!(rec::Recommender; kwargs...)
    error("build! is not implemented for recommender type $(typeof(rec))")
end

function recommend(rec::Recommender, u::Int, k::Int, candidates::Array{Int})
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
    check_build_status(rec)
    predict(rec, u, i)
end

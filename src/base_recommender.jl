export Recommender, ContentRecommender
export build, execute, predict, ranking

abstract Recommender

function build(rec::Recommender; kwargs...)
    error("build is not implemented for recommender type $(typeof(rec))")
end

function execute{T}(rec::Recommender, u::Int, k::Int, item_names::AbstractVector{T})
    d = Dict{T,Float64}()
    n_item = size(item_names, 1)
    for i in 1:n_item
        score = ranking(rec, u, i)
        d[item_names[i]] = score
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

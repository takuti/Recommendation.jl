export Recommender, ContentRecommender
export build, execute, predict, ranking

abstract Recommender

function build(recommender::Recommender; kwargs...)
    error("build is not implemented for recommender type $(typeof(recommender))")
end

function execute{T}(recommender::Recommender, u::Int, k::Int, item_names::AbstractVector{T})
    d = Dict{T,Float64}()
    n_item = size(item_names, 1)
    for i in 1:n_item
        score = ranking(recommender, u, i)
        d[item_names[i]] = score
    end
    sort(collect(d), by=tuple->last(tuple), rev=true)[1:k]
end

function predict(recommender::Recommender, u::Int, i::Int)
    error("predict is not implemented for recommender type $(typeof(recommender))")
end

# Return a ranking score of item i for user u
function ranking(recommender::Recommender, u::Int, i::Int)
    predict(recommender, u, i)
end

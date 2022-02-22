import Base: depwarn

export build!

function build!(recommender::Recommender; kwargs...)
    depwarn("`build!`` is deprecated and renamed to `fit!`", :build!)
    fit!(recommender; kwargs...)
end

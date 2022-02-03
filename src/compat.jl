export build!

function build!(recommender::Recommender; kwargs...)
    @warn "`build!`` is deprecated and renamed to `fit!`"
    fit!(recommender; kwargs...)
end

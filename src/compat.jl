export build!

function bridge_fit!(recommender::Recommender; kwargs...)
    @warn "`build!`` is deprecated and renamed to `fit!`"
    fit!(recommender; kwargs...)
end

build!(recommender::Recommender; kwargs...) = bridge_fit!(recommender; kwargs...)

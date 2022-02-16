export SyntheticFeature, SyntheticRule, accumulate, generate

struct SyntheticFeature
    name::String
    candidates::AbstractVector{Union{String, Infinite}}
    value::Union{Unknown, String, Infinite}

    function SyntheticFeature(name::String, candidates::UnitRange)
        new(name, collect(candidates), missing)
    end

    function SyntheticFeature(name::String, candidates::AbstractVector)
        new(name, candidates, missing)
    end

    function SyntheticFeature(name::String, candidates::AbstractVector, value::Union{Unknown, String, Infinite})
        new(name, candidates, value)
    end
end

Random.rand(rng::AbstractRNG, f::Random.SamplerTrivial{SyntheticFeature}) =
    SyntheticFeature(f[].name, f[].candidates, rand(rng, f[].candidates))

struct SyntheticRule
    # item ID
    item::Union{Nothing, Integer}

    # return bool
    match::Function

    # if f returns true, accumulative CTR is lifted p%
    probability::Float64
end

function accumulate(item::Integer, sample::Dict{String, Any}, rules::AbstractVector{SyntheticRule})
    cumulative_probability = 0.0
    for rule in rules
        if (isnothing(rule.item) || rule.item == item) && rule.match(sample)
            cumulative_probability += rule.probability
        end
    end
    cumulative_probability
end

function generate(n_samples::Integer, n_items::Integer, features::AbstractVector{SyntheticFeature}, rules::AbstractVector{SyntheticRule})
    samples = hcat(map(f ->  rand(f, n_samples), features)...) # n_samples * len(features) matrix
    feedback = Bool[]
    for sample in eachrow(samples)
        push!(
            feedback,
            rand() <= accumulate(rand(1:n_items), Dict(map(f -> (f.name, f.value), sample)), rules)
        )
    end
    feedback
end

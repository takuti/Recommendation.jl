export SyntheticFeature, SyntheticRule, accumulate, generate

struct SyntheticFeature
    name::String
    candidates::Union{UnitRange, AbstractVector}
    value::Union{Unknown, String, Infinite}

    function SyntheticFeature(name::String, candidates::Union{UnitRange, AbstractVector})
        value_type = eltype(candidates)
        if value_type <: Infinite || value_type <: String
            new(name, candidates, missing)
        else
            error("unsupported feature value type: $value_type")
        end
    end

    function SyntheticFeature(name::String, candidates::Union{UnitRange, AbstractVector}, value::Union{Unknown, String, Infinite})
        new(name, candidates, value)
    end
end

to_dict(features::AbstractVector{SyntheticFeature}) = Dict(map(feature -> (feature.name, feature.value), features))

Random.rand(rng::AbstractRNG, f::Random.SamplerTrivial{SyntheticFeature}) =
    SyntheticFeature(f[].name, f[].candidates, rand(rng, f[].candidates))

struct SyntheticRule
    # if f returns true, accumulative CTR is lifted p%
    probability::Float64

    # item ID
    item::Union{Nothing, Integer}

    # return bool
    match::Function

    function SyntheticRule(probability::Float64)
        # matching any items, any combinations of features
        new(probability, nothing, _ -> true)
    end

    function SyntheticRule(probability::Float64, item::Union{Nothing, Integer})
        # matching any combinations of features
        new(probability, item, _ -> true)
    end

    function SyntheticRule(probability::Float64, item::Union{Nothing, Integer}, match::Function)
        new(probability, item, match)
    end
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
            rand() <= accumulate(rand(1:n_items), to_dict(sample), rules)
        )
    end
    feedback
end

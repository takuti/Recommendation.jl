export SyntheticFeature, SyntheticRule, accumulate, generate

struct SyntheticFeature
    name::String
    candidates::AbstractVector{Union{String, Infinite}}
    value::Union{String, Infinite}

    function SyntheticFeature(name::String, candidates::UnitRange)
        new(name, collect(candidates), -1)
    end

    function SyntheticFeature(name::String, candidates::AbstractVector)
        new(name, candidates, -1)
    end

    function SyntheticFeature(name::String, candidates::AbstractVector, value::Union{String, Infinite})
        new(name, candidates, value)
    end
end

Random.rand(rng::AbstractRNG, f::Random.SamplerTrivial{SyntheticFeature}) =
    SyntheticFeature(f[].name, f[].candidates, rand(rng, f[].candidates))

struct SyntheticRule
    # return bool
    is_match::Function

    # if f returns true, accumulative CTR is lifted p%
    prob::Float64
end

function accumulate(sample::Dict{String, Any}, rules::AbstractVector{SyntheticRule})
    ctr = 0.0
    for rule in rules
        if rule.is_match(sample)
            ctr += rule.prob
        end
    end
    ctr
end

function generate(n_samples::Integer, features::AbstractVector{SyntheticFeature}, rules::AbstractVector{SyntheticRule})
    samples = hcat(map(f ->  rand(f, n_samples), features)...) # n_samples * len(features) matrix
    feedback = Bool[]
    for sample in eachrow(samples)
        push!(
            feedback,
            rand() <= accumulate(Dict(map(f -> (f.name, f.value), sample)), rules)
        )
    end
    feedback
end

export SyntheticFeature, SyntheticRule, accumulate, generate

"""
    SyntheticFeature(name::String, candidates::Union{UnitRange, AbstractVector})

Synthetic feature generator that allows us to sample a value from `candidates`.

```julia
features = SyntheticFeature[]

age = SyntheticFeature("Age", 1930:2010))
push!(features, age)

geo = SyntheticFeature("Geo", ["Arizona", "California", "Colorado", "Illinois", "Indiana", "Michigan", "New York", "Utah"])
push!(features, geo)

rand(geo, 3) # e.g., ["California", "New York", "Arizona"]
```
"""
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

vectorize(feature::SyntheticFeature) = (feature.candidates isa UnitRange) ? [feature.value] : onehot(feature.value, feature.candidates)
to_dict(features::AbstractVector{SyntheticFeature}) = Dict(map(feature -> (feature.name, feature.value), features))

Random.rand(rng::AbstractRNG, f::Random.SamplerTrivial{SyntheticFeature}) =
    SyntheticFeature(f[].name, f[].candidates, rand(rng, f[].candidates))

"""
    SyntheticRule{probability::Float64[, item::Union{Nothing, Integer}, match::Function])

Matching rule for cumulative "click through rate". Given an item index,
we increase the probability of acceptance upon sampling when `match` returns `true`,
which takes a dictionary of feature name => value.

```julia
rules = SyntheticRule[]

push!(rules, SyntheticRule(0.001))
push!(rules, SyntheticRule(0.01, 3))
push!(rules, SyntheticRule(0.30, 1, s -> s["Age"] >= 1980 && s["Age"] <= 1989 && s["Geo"] == "New York"))
push!(rules, SyntheticRule(0.30, 2, s -> s["Age"] >= 1950 && s["Age"] <= 1959 && s["Geo"] == "New York"))
push!(rules, SyntheticRule(0.30, 2, s -> s["Age"] >= 1980 && s["Age"] <= 1989 && s["Geo"] == "Arizona"))
push!(rules, SyntheticRule(0.30, 1, s -> s["Age"] >= 1950 && s["Age"] <= 1959 && s["Geo"] == "Arizona"))
```
"""
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

"""
    generate(n_samples::Integer, n_items::Integer, features::AbstractVector{SyntheticFeature}, rules::AbstractVector{SyntheticRule}) -> DataAccessor

Generate a synthetic data accessor from randomly sampled implicit feedback.
Each sample is considered as a different user, and user attributes are represented by
a numeric onehot-encoded feature vector based on the values returned by `SyntheticFeature`.

The process is based on Section 7.3 of the following paper:

- M. Aharon, et al.
  **OFF-Set: One-pass Factorization of Feature Sets for Online Recommendation in Persistent Cold Start Settings**.
  [arXiv:1308.1792](https://arxiv.org/abs/1308.1792).

```julia
n_samples = 256
n_items = 5
data = generate(n_samples, n_items, features, rules)
```
"""
function generate(n_samples::Integer, n_items::Integer, features::AbstractVector{SyntheticFeature}, rules::AbstractVector{SyntheticRule})
    events = Event[]
    attributes = Dict()

    samples = hcat(map(f ->  rand(f, n_samples), features)...) # n_samples * len(features) matrix
    for (user, sample) in enumerate(eachrow(samples))
        item = rand(1:n_items)
        if rand() <= accumulate(item, to_dict(sample), rules)
            push!(events, Event(user, item, 1.0))  # record the probabilistically accepted binary feedback
            attributes[user] = collect(Iterators.flatten(map(vectorize, sample)))
        end
    end

    data = DataAccessor(events, n_samples, n_items)
    for (user, user_attribute) in attributes
        set_user_attribute(data, user, user_attribute)
    end

    data
end

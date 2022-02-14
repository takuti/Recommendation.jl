println("-- Testing synthetic implicit feedback generator")

# define a set of features
features = Feature[]

push!(features, Feature("Age", 1950, () -> rand(1930:2010)))
push!(features, Feature("Geo", "Arizona", () -> rand(["Arizona", "California", "Colorado", "Illinois", "Indiana", "Michigan", "New York", "Utah"])))
push!(features, Feature("Ad", 0, () -> rand(0:4)))

# create a set of rules
rules = Rule[]

push!(rules, Rule(s -> true, 0.001))
push!(rules, Rule(s -> s["Ad"] == 2, 0.01))
push!(rules, Rule(s -> s["Age"] >= 1980 && s["Age"] <= 1989 && s["Geo"] == "New York" && s["Ad"] == 0, 0.30))
push!(rules, Rule(s -> s["Age"] >= 1950 && s["Age"] <= 1959 && s["Geo"] == "New York" && s["Ad"] == 1, 0.30))
push!(rules, Rule(s -> s["Age"] >= 1980 && s["Age"] <= 1989 && s["Geo"] == "Arizona" && s["Ad"] == 1, 0.30))
push!(rules, Rule(s -> s["Age"] >= 1950 && s["Age"] <= 1959 && s["Geo"] == "Arizona" && s["Ad"] == 0, 0.30))

# check if a accumulative CTR is computed correctly
sample1 = Dict("Age" => 1940, "Geo" => "California", "Ad" => 0)
@test Recommendation.accumulate(sample1, rules) == 0.001
@test isa(generate(sample1, rules), Bool)

sample2 = Dict("Age" => 1940, "Geo" => "California", "Ad" => 2)
@test Recommendation.accumulate(sample2, rules) == 0.011
@test isa(generate(sample2, rules), Bool)

sample3 = Dict("Age" => 1953, "Geo" => "New York", "Ad" => 1)
@test Recommendation.accumulate(sample3, rules) == 0.301
@test isa(generate(sample3, rules), Bool)

# generate samples with random pairs of demographics and ad variants
samples = Dict[]

for i in 1:100
    sample = Dict()
    for f in features
        sample[f.name] = f.random()
    end
    push!(samples, sample)
end

feedback = generate(samples, rules)

@test isa(feedback, Array{Bool})
@test length(feedback) == length(samples)

println("-- Testing synthetic implicit feedback generator")

# define a set of features
features = SyntheticFeature[]

push!(features, SyntheticFeature("Age", 1930:2010))
push!(features, SyntheticFeature("Geo", ["Arizona", "California", "Colorado", "Illinois", "Indiana", "Michigan", "New York", "Utah"]))
push!(features, SyntheticFeature("Ad", 0:4))

# create a set of rules
rules = SyntheticRule[]

push!(rules, SyntheticRule(s -> true, 0.001))
push!(rules, SyntheticRule(s -> s["Ad"] == 2, 0.01))
push!(rules, SyntheticRule(s -> s["Age"] >= 1980 && s["Age"] <= 1989 && s["Geo"] == "New York" && s["Ad"] == 0, 0.30))
push!(rules, SyntheticRule(s -> s["Age"] >= 1950 && s["Age"] <= 1959 && s["Geo"] == "New York" && s["Ad"] == 1, 0.30))
push!(rules, SyntheticRule(s -> s["Age"] >= 1980 && s["Age"] <= 1989 && s["Geo"] == "Arizona" && s["Ad"] == 1, 0.30))
push!(rules, SyntheticRule(s -> s["Age"] >= 1950 && s["Age"] <= 1959 && s["Geo"] == "Arizona" && s["Ad"] == 0, 0.30))

# check if a accumulative CTR is computed correctly
sample1 = Dict("Age" => 1940, "Geo" => "California", "Ad" => 0)
@test Recommendation.accumulate(sample1, rules) == 0.001

sample2 = Dict("Age" => 1940, "Geo" => "California", "Ad" => 2)
@test Recommendation.accumulate(sample2, rules) == 0.011

sample3 = Dict("Age" => 1953, "Geo" => "New York", "Ad" => 1)
@test Recommendation.accumulate(sample3, rules) == 0.301

# generate samples with random pairs of demographics and ad variants
n_samples = 100
feedback = generate(n_samples, features, rules)

@test isa(feedback, Array{Bool})
@test length(feedback) == n_samples

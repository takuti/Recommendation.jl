println("-- Testing synthetic implicit feedback generator")

# define a set of features
features = SyntheticFeature[]

push!(features, SyntheticFeature("Age", 1930:2010))
push!(features, SyntheticFeature("Geo", ["Arizona", "California", "Colorado", "Illinois", "Indiana", "Michigan", "New York", "Utah"]))
@test_throws ErrorException SyntheticFeature("Foo", [1, 2, 1+2im])
@test_throws ErrorException SyntheticFeature("Foo", ['a', 'b'])

# create a set of rules
rules = SyntheticRule[]

push!(rules, SyntheticRule(0.001))
push!(rules, SyntheticRule(0.01, 3))
push!(rules, SyntheticRule(0.30, 1, s -> s["Age"] >= 1980 && s["Age"] <= 1989 && s["Geo"] == "New York"))
push!(rules, SyntheticRule(0.30, 2, s -> s["Age"] >= 1950 && s["Age"] <= 1959 && s["Geo"] == "New York"))
push!(rules, SyntheticRule(0.30, 2, s -> s["Age"] >= 1980 && s["Age"] <= 1989 && s["Geo"] == "Arizona"))
push!(rules, SyntheticRule(0.30, 1, s -> s["Age"] >= 1950 && s["Age"] <= 1959 && s["Geo"] == "Arizona"))

# check if a accumulative CTR is computed correctly
sample1 = Dict("Age" => 1940, "Geo" => "California")
@test Recommendation.accumulate(1, sample1, rules) == 0.001

sample2 = Dict("Age" => 1940, "Geo" => "California")
@test Recommendation.accumulate(3, sample2, rules) == 0.011

sample3 = Dict("Age" => 1953, "Geo" => "New York")
@test Recommendation.accumulate(2, sample3, rules) == 0.301

# generate samples with random pairs of demographics and ad variants
n_samples = 256
n_items = 5
data = generate(n_samples, n_items, features, rules)

@test isa(data, DataAccessor)
@test size(data.R) == (n_samples, n_items)
@test length(data.events) == length(data.user_attributes)
for (user, user_attribute) in data.user_attributes
    @test all(map(value -> typeof(value) <: Infinite, user_attribute))
end
@test isempty(data.item_attributes)

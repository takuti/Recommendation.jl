# If this script is executed directly from the Recommendation.jl repository,
# instantiate the package from the local code so everything is up-to-date.
# Otherwise, we assume Recommendation.jl is pre-installed, and
# `using Recommendation` works without manual activation.
proj_toml_path = normpath(joinpath(@__DIR__, "..", "..", "Project.toml"))
if isfile(proj_toml_path)
    using TOML
    conf = TOML.parse(read(proj_toml_path, String))
    if conf["name"] == "Recommendation"
        using Pkg
        Pkg.activate(dirname(proj_toml_path))
        Pkg.instantiate()
    end
end

using Recommendation

# configureable parameters for cross validation
n_fold = 2
metric = Recall
topk = 5
recommender = MostPopular
data = load_movielens_100k()

res = cross_validation(n_fold, metric, topk,
                       recommender, data)
println("Average $metric = $res")

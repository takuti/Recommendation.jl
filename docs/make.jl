using Documenter, Recommendation

makedocs(
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true"
    ),
    modules = [Recommendation],
    sitename = "Recommendation.jl",
    authors = "Takuya Kitazawa",
    linkcheck = !("skiplinks" in ARGS),
    pages = [
        "Home" => "index.md",
        "getting_started.md",
        "data.md",
        "notation.md",
        "baseline.md",
        "collaborative_filtering.md",
        "factorization_machines.md",
        "content_based_filtering.md",
        "evaluation.md",
    ],
)

deploydocs(
    repo = "github.com/takuti/Recommendation.jl.git",
    target = "build",
)

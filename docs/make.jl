using Documenter, Recommendation

makedocs(
    format = :html,
    modules = [Recommendation],
    sitename = "Recommendation.jl",
    authors = "Takuya Kitazawa",
    linkcheck = !("skiplinks" in ARGS),
    pages = [
        "Home" => "index.md",
        "Getting Started" => "getting_started.md",
        "Reference" => [
            "baseline.md",
            "cf.md",
            "content_based.md",
            "evaluation.md",
        ],
    ],
)

deploydocs(
    repo = "github.com/takuti/Recommendation.jl.git",
    target = "build",
)

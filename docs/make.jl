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
        "References" => [
            "notation.md",
            "baseline.md",
            "collaborative_filtering.md",
            "content_based_filtering.md",
            "evaluation.md",
        ],
    ],
)

deploydocs(
    repo = "github.com/takuti/Recommendation.jl.git",
    target = "build",
)

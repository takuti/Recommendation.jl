using Documenter, Recommendation

makedocs(
    format = :html,
    modules = [Recommendation],
    sitename = "Recommendation.jl",
    authors = "Takuya Kitazawa",
    linkcheck = !("skiplinks" in ARGS),
    pages = [
        "Home" => "index.md",
        "getting_started.md",
    ],
)

deploydocs(
    repo = "github.com/takuti/Recommendation.jl.git",
    target = "build",
)

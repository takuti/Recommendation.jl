export get_data_home, download_file, load_movielens_100k

# function download_data()

"""
    get_data_home([data_home=nothing]) -> String

Return an absolute path to a directory containing datasets. Create the directory if it does not exist, and
`data_home=nothing` defaults to either an environmental variable `JULIA_RECOMMENDATION_DATA` or `~/julia_recommendation_data`.

Reference: [Similar function in scikit-learn](https://github.com/scikit-learn/scikit-learn/blob/7e1e6d09bcc2eaeba98f7e737aac2ac782f0e5f1/sklearn/datasets/_base.py#L34).
"""
function get_data_home(data_home::Union{String, Nothing}=nothing)
    if data_home == nothing
        data_home = get(ENV, "JULIA_RECOMMENDATION_DATA", joinpath("~", "julia_recommendation_data"))
    end
    data_home = expanduser(data_home)
    mkpath(data_home)
end


"""
    download_file(url, path=nothing) -> path

Download a dataset from the URL to the path. Create folders if needed. `path=nothing` defaults to `tempname()` as a destination.
"""
function download_file(url::String, path::Union{String, Nothing}=nothing)
    if path == nothing
        path = tempname()
    end
    if isfile(path)
        error("file already exists: $path")
    end
    data_home = get_data_home(dirname(path))  # ensure the directory exists
    download(url, joinpath(data_home, basename(path)))
end


"""
    load_movielens_100k([path=nothing]) -> DataAccessor

Read user-item-rating triples from a locally saved `u.data` TSV file downloaded from
[MovieLens 100k](https://grouplens.org/datasets/movielens/100k/), and convert them into a `DataAccessor` instance.

Download if `path` is not given or the specified file does not exist.
"""
function load_movielens_100k(path::Union{String, Nothing}=nothing)
    n_user = 943
    n_item = 1682

    if path == nothing || !isfile(path)
        path = download_file("https://files.grouplens.org/datasets/movielens/ml-100k/u.data", path)
    end

    R = matrix(n_user, n_item)
    open(path, "r") do f
        for line in eachline(f)
            l = split(line, "\t")
            user, item, value = parse(Int, l[1]), parse(Int, l[2]), parse(Int, l[3])
            R[user, item] = value
        end
    end
    DataAccessor(R)
end

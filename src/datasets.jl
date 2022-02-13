export get_data_home, download_file, unzip, load_movielens_100k, load_amazon_review

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
        path = joinpath(tempname(), basename(url))
    end
    if isfile(path)
        @warn "file already exists, so skip downloading: $path"
        path
    else
        data_home = get_data_home(dirname(path))  # ensure the directory exists
        Downloads.download(url, joinpath(data_home, basename(path)))
    end
end


"""
    unzip(path[, exdir=nothing]) -> exdir

Extract files in a zip file at `path` into a directory `exdir`.
Extract into the same directory as the zip file if `exdir=nothing`.

Reference: https://github.com/fhs/ZipFile.jl/pull/16
"""
function unzip(path::String, exdir::Union{String, Nothing}=nothing)
    if exdir == nothing
        exdir = dirname(path)
    end
    zip_reader = ZipFile.Reader(path)
    for file in zip_reader.files
        out_path = joinpath(exdir, file.name)
        if isdirpath(out_path)
            mkpath(out_path)
        else
            open(out_path, "w") do io
                write(io, read(file, String))
            end
        end
    end
    close(zip_reader)
    exdir
end


"""
    load_movielens_100k([path=nothing]) -> DataAccessor

`path` points to a locally saved [MovieLens 100k](https://grouplens.org/datasets/movielens/100k/).
Read user-item-rating triples in the folder, and convert them into a `DataAccessor` instance.

Download and decompress a corresponding zip file, if `path` is not given or the specified folder does not exist.
"""
function load_movielens_100k(path::Union{String, Nothing}=nothing)
    n_user = 943
    n_item = 1682
    R = matrix(n_user, n_item)

    if path == nothing || !isdir(path)
        zip_path = path
        if zip_path != nothing
            zip_path = joinpath(dirname(zip_path), "ml-100k.zip")
        end
        zip_path = download_file("https://files.grouplens.org/datasets/movielens/ml-100k.zip", zip_path)
        path = joinpath(unzip(zip_path, path), "ml-100k/")
    end

    open(joinpath(path, "u.data"), "r") do io
        for line in eachline(io)
            l = split(line, "\t")
            user, item, value = parse(Int, l[1]), parse(Int, l[2]), parse(Int, l[3])
            R[user, item] = value
        end
    end
    data = DataAccessor(R)

    gender_set = ["M", "F"]
    occupation_set = []
    open(joinpath(path, "u.occupation"), "r") do io
        for line in eachline(io)
            push!(occupation_set, String(line))
        end
    end

    open(joinpath(path, "u.user"), "r") do io
        for line in eachline(io)
            l = split(line, "|")
            user, age, gender, occupation = parse(Int, l[1]), parse(Int, l[2]), String(l[3]), String(l[4])
            set_user_attribute(data, user, [age, onehot(gender, gender_set)..., onehot(occupation, occupation_set)...])
        end
    end

    open(joinpath(path, "u.item"), "r") do io
        for line in eachline(io)
            l = split(line, "|")
            item = parse(Int, l[1])
            genres = map(s -> parse(Float64, s), last(l, 19)) # last 19 fields are genres (already onehot-encoded)
            set_item_attribute(data, item, genres)
        end
    end

    data
end


"""
    load_amazon_review([path=nothing; category="Electronics"]) -> DataAccessor

`path` points to a locally saved small set of [Amazon Reviews Dataset](https://nijianmo.github.io/amazon/index.html)
for a particular category. Each row has a tuple of (item, user, rating, timestamp).
"""
function load_amazon_review(path::Union{String, Nothing}=nothing; category::String="Electronics")
    categories = Set([
        "AMAZON_FASHION",
        "All_Beauty",
        "Appliances",
        "Arts_Crafts_and_Sewing",
        "Automotive",
        "Books",
        "CDs_and_Vinyl",
        "Cell_Phones_and_Accessories",
        "Clothing_Shoes_and_Jewelry",
        "Digital_Music",
        "Electronics",
        "Gift_Cards",
        "Grocery_and_Gourmet_Food",
        "Home_and_Kitchen",
        "Industrial_and_Scientific",
        "Kindle_Store",
        "Luxury_Beauty",
        "Magazine_Subscriptions",
        "Movies_and_TV",
        "Musical_Instruments",
        "Office_Products",
        "Patio_Lawn_and_Garden",
        "Pet_Supplies",
        "Prime_Pantry",
        "Software",
        "Sports_and_Outdoors",
        "Tools_and_Home_Improvement",
        "Toys_and_Games",
        "Video_Games"
    ])
    if category ∉ categories
        error("category $category does not exist.")
    end
    if path == nothing || !isfile(path)
        path = download_file("http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/$category.csv", path)
    end
    events = Array{Event, 1}()
    n_user, n_item = 0, 0
    user_ids, item_ids = Dict{String, Integer}(), Dict{String, Integer}()
    open(path, "r") do io
        for line in eachline(io)
            l = split(line, ",")
            item, user, rating = String(l[1]), String(l[2]), parse(Float64, l[3])
            if haskey(user_ids, user)
                u = user_ids[user]
            else
                n_user += 1
                u = n_user
                user_ids[user] = n_user
            end
            if haskey(item_ids, item)
                i = item_ids[item]
            else
                n_item += 1
                i = n_item
                item_ids[item] = n_item
            end
            push!(events, Event(u, i, rating))
        end
    end
    DataAccessor(events, n_user, n_item)
end

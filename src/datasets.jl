export load_movielens_100k

"""
    load_movielens_100k(path::String)

Read user-item-rating triples from a locally saved `u.data` TSV file downloaded from [MovieLens 100k](https://grouplens.org/datasets/movielens/100k/), and convert them into a `DataAccessor` instance.
"""
function load_movielens_100k(path::String)
    n_user = 943
    n_item = 1682

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

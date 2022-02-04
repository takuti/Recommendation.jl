export load_movielens_100k

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
    R
end

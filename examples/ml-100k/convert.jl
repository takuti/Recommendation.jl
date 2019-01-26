using JLD
using SparseArrays

const n_user = 943
const n_item = 1682

R = spzeros(n_user, n_item)

open("ml-100k/u.data", "r") do f
    for line in eachline(f)
        l = split(line, "\t")
        user, item, value = parse(Int, l[1]), parse(Int, l[2]), parse(Int, l[3])
        R[user, item] = value
    end
end

save("ml-100k.jld", "R", R)

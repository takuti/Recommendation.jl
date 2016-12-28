using JLD

const n_user = 6040
const n_item = 3952

R = spzeros(n_user, n_item)

open("ml-1m/ratings.dat", "r") do f
    for line in eachline(f)
        l = split(line, "::")
        user, item, value = parse(Int, l[1]), parse(Int, l[2]), parse(Int, l[3])
        R[user, item] = value
    end
end

save("ml-1m.jld", "R", R)

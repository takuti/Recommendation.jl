export matrix, vector, isfilled, onehot

function matrix(m::Integer, n::Integer)
    Array{Union{Missing, AbstractFloat}}(missing, m, n)
end

function vector(m::Integer)
    Array{Union{Missing, AbstractFloat}}(missing, m)
end

function isfilled(a::AbstractArray)
    findfirst(v -> isa(v, Unknown), a) == nothing
end

function onehot(value::Union{Unknown, Integer, String}, value_set::AbstractVector)
    if !allunique(value_set)
        error("duplicated value exists in a value set")
    end
    filter!(val -> !isa(val, Unknown), value_set)
    dims = length(value_set)
    if isa(value, Unknown)
        zeros(dims)
    else
        idx = findfirst(isequal(value), value_set)
        if idx == nothing
            error("value not found")
        end
        vec = zeros(dims)
        vec[idx] = 1.0
        vec
    end
end

function onehot(vec::AbstractVector)
    value_set = unique(vec)
    vcat(map(value -> onehot(value, value_set)', vec)...)
end

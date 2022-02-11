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

"""
    onehot(value, value_set::AbstractVector) -> Vector{Float64}

Encode a categorical value to a onehot-encoded vector.
Value must be one of the elements in `value_set` in `Integer` or `String` type.
`missing` or `nothing` are also acceptable as a value, but they are converted into a zero vector.
"""
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

"""
    onehot(vec::AbstractVector) -> Matrix{Float64}

Encode a categorical vector to a onehot-encoded matrix.
`["Male", "Female", "Others"]` is converted into `[1. 0. 0.; 0. 1. 0.; 0. 0. 1.]`.
An index corresponding to a possible value is assigned in the order of first-time appearance in the input vector.
"""
function onehot(vec::AbstractVector)
    value_set = unique(vec)
    vcat(map(value -> onehot(value, value_set)', vec)...)
end

"""
    onehot(mat::AbstractMatrix) -> Matrix{Float64}

Each column of an input matrix represents a single categorical vector.
Onehot-encode the individual columns and horizontally concatenate them as an output.
"""
function onehot(mat::AbstractMatrix)
    hcat(map(onehot, eachcol(mat))...)
end

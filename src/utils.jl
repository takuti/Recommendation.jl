export matrix, vector, isfilled, get_pairwise_preference_triples, onehot, binarize_multi_label

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
    get_pairwise_preference_triples(R::AbstractMatrix) -> Vector{Tuple{Int, Int, Int}}

Return user-item-item triples corresponding to a user-item matrix `R`
(i.e., ``(u, i, j) \\in D_s`` in [BPR: Bayesian Personalized Ranking from Implicit Feedback](https://dl.acm.org/doi/10.5555/1795114.1795167)).
In the pairwise item ranking context, each triple represents that user ``u`` prefers item ``i`` over ``j``.
"""
function get_pairwise_preference_triples(R::AbstractMatrix)
    vcat(map(t -> vcat(collect(Iterators.product(t...))...),
             filter(t -> length(t[2]) > 0 && length(t[3]) > 0,
                    map(t -> ([t[1]], findall(!iszero, t[2]), findall(iszero, t[2])),
                        enumerate(eachrow(R)))))...)
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

function binarize_multi_label(values::AbstractVector, value_set::AbstractVector)
    if !allunique(value_set)
        error("duplicated value exists in a value set")
    end
    filter!(val -> !isa(val, Unknown), values)
    filter!(val -> !isa(val, Unknown), value_set)
    dims = length(value_set)
    indices = findall(v -> v in values, value_set)
    vec = zeros(dims)
    vec[indices] .= 1.0
    vec
end

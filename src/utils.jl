export Event, matrix, vector, isfilled, almost_zero, isalmostzero

mutable struct Event
    user::Int
    item::Int
    value::Float64 # e.g. rating, 0/1
end

accepted_types = Union{Nothing, Missing, Number}

function matrix(m::Int, n::Int;
                type::Type{<:accepted_types}=Union{Nothing, Float64}, initializer=nothing)
    Array{type}(initializer, m, n)
end

function vector(m::Int;
                type::Type{<:accepted_types}=Union{Nothing, Float64}, initializer=nothing)
    Array{type}(initializer, m)
end

function isfilled(a::AbstractArray; by_value=nothing)
    by_value ∉ Set(a)
end

almost_zero = 1e-256 # including `undef`

function isalmostzero(x::Number)
    x <= almost_zero
end

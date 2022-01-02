export Event, matrix, vector, isfilled, almost_zero, isalmostzero

mutable struct Event
    user::Integer
    item::Integer
    value::AbstractFloat # e.g. rating, 0/1
end

accepted_types = Union{Nothing, Missing, AbstractFloat, Integer}

function matrix(m::Integer, n::Integer;
                type::Type{<:accepted_types}=Union{Nothing, AbstractFloat}, initializer=nothing)
    Array{type}(initializer, m, n)
end

function vector(m::Integer;
                type::Type{<:accepted_types}=Union{Nothing, AbstractFloat}, initializer=nothing)
    Array{type}(initializer, m)
end

function isfilled(a::AbstractArray; by_value=nothing)
    by_value ∉ Set(a)
end

almost_zero = 1e-256

function isalmostzero(x::Number)
    x <= almost_zero
end

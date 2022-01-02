export matrix, vector, isfilled, almost_zero, isalmostzero

function matrix(m::Integer, n::Integer;
                type::Type{<:DataValue}=Union{Nothing, AbstractFloat}, initializer=nothing)
    Array{type}(initializer, m, n)
end

function vector(m::Integer;
                type::Type{<:DataValue}=Union{Nothing, AbstractFloat}, initializer=nothing)
    Array{type}(initializer, m)
end

function isfilled(a::AbstractArray; by_value=nothing)
    by_value ∉ Set(a)
end

almost_zero = 1e-256

function isalmostzero(x::Number)
    x <= almost_zero
end

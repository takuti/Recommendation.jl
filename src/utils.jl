export Event, matrix, vector, isfilled

mutable struct Event
    user::Int
    item::Int
    value::Float64 # e.g. rating, 0/1
end

function matrix(m::Int, n::Int, t::Type{<:Number}=Float64)
    Array{Union{Nothing, t}}(nothing, m, n)
end

function vector(m::Int, t::Type{<:Number}=Float64)
    Array{Union{Nothing, t}}(nothing, m)
end

function isfilled(a::AbstractArray)
    nothing ∉ Set(a)
end

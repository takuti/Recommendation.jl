export Event, matrix, vector, isfilled

mutable struct Event
    user::Int
    item::Int
    value::Float64 # e.g. rating, 0/1
end

function matrix(m::Int, n::Int)
    Array{Union{Nothing, Float64}}(nothing, m, n)
end

function vector(m::Int)
    Array{Union{Nothing, Float64}}(nothing, m)
end

function isfilled(a::AbstractArray)
    nothing ∉ Set(a)
end

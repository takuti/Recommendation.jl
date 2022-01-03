export matrix, vector, isfilled

function matrix(m::Integer, n::Integer)
    Array{Union{Missing, AbstractFloat}}(missing, m, n)
end

function vector(m::Integer)
    Array{Union{Missing, AbstractFloat}}(missing, m)
end

function isfilled(a::AbstractArray)
    findfirst(v -> isa(v, Unknown), a) == nothing
end

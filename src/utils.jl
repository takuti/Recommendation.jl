export matrix, vector, isfilled

function matrix(m::Integer, n::Integer)
    Array{Union{Missing, AbstractFloat}}(missing, m, n)
end

function vector(m::Integer)
    Array{Union{Missing, AbstractFloat}}(missing, m)
end

function isfilled(a::AbstractArray)
    missing ∉ Set(a)
end

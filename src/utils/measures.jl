export Measures

module Measures

function count_true_positive{T}(truth::Array{T}, recommend::Array{T})
    tp = 0
    for r in recommend
        if findfirst(truth, r) != 0
            tp += 1
        end
    end
    tp
end

function recall{T}(truth::Array{T}, recommend::Array{T}, k::Int)
    count_true_positive(truth, recommend[1:k]) / length(truth)
end

end # module Measures

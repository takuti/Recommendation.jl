export AccuracyMetric

module AccuracyMetric

# Root Mean Squared Error
function rmse(truth::AbstractVector, prediction::AbstractVector)
    sqrt(sum((truth - prediction).^2) / length(truth))
end

# Mean Absolute Error
function mae(truth::AbstractVector, prediction::AbstractVector)
    sum(abs(truth - prediction)) / length(truth)
end

end # module AccuracyMetric

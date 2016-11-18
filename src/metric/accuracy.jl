export RMSE, MAE

# Root Mean Squared Error
immutable RMSE <: AccuracyMetric end
function measure(metric::RMSE, truth::AbstractVector, prediction::AbstractVector)
    sqrt(sum((truth - prediction).^2) / length(truth))
end

# Mean Absolute Error
immutable MAE <: AccuracyMetric end
function measure(metric::MAE, truth::AbstractVector, prediction::AbstractVector)
    sum(abs(truth - prediction)) / length(truth)
end

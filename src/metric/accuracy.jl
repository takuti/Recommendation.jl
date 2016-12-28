export RMSE, MAE

# Root Mean Squared Error
immutable RMSE <: AccuracyMetric end
function measure(metric::RMSE, truth::AbstractVector, pred::AbstractVector)
    sqrt(sum((truth - pred).^2) / length(truth))
end

# Mean Absolute Error
immutable MAE <: AccuracyMetric end
function measure(metric::MAE, truth::AbstractVector, pred::AbstractVector)
    sum(abs.(truth - pred)) / length(truth)
end

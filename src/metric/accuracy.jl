export RMSE, MAE

"""
    RMSE

Root Mean Squared Error.

    measure(
        metric::RMSE,
        truth::AbstractVector,
        pred::AbstractVector
    )
"""
struct RMSE <: AccuracyMetric end
function measure(metric::RMSE, truth::AbstractVector, pred::AbstractVector)
    sqrt(sum((truth - pred).^2) / length(truth))
end

"""
    MAE

Mean Absolute Error.

    measure(
        metric::MAE,
        truth::AbstractVector,
        pred::AbstractVector
    )
"""
struct MAE <: AccuracyMetric end
function measure(metric::MAE, truth::AbstractVector, pred::AbstractVector)
    sum(abs.(truth - pred)) / length(truth)
end

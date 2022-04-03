export RMSE, MAE

"""
    RMSE

Root Mean Squared Error.

    measure(
        metric::RMSE,
        truth::AbstractVector,
        pred::AbstractVector
    )

`truth` and `pred` are expected to be the same size.
"""
struct RMSE <: AccuracyMetric end
function measure(metric::RMSE, truth::AbstractVector, pred::AbstractVector)
    n = length(truth)
    if n != length(pred)
        error("`truth` and `pred` have different size, which are $(n) and $(length(pred)), respectively")
    end
    if n == 0
        0.0
    else
        sqrt(sum((truth - pred).^2) / n)
    end
end

"""
    MAE

Mean Absolute Error.

    measure(
        metric::MAE,
        truth::AbstractVector,
        pred::AbstractVector
    )

`truth` and `pred` are expected to be the same size.
"""
struct MAE <: AccuracyMetric end
function measure(metric::MAE, truth::AbstractVector, pred::AbstractVector)
    n = length(truth)
    if n != length(pred)
        error("`truth` and `pred` have different size, which are $(n) and $(length(pred)), respectively")
    end
    if n == 0
        0.0
    else
        sum(abs.(truth - pred)) / n
    end
end

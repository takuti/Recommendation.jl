export cross_validation, leave_one_out

"""
    cross_validation(
        n_folds::Integer,
        metric::RankingMetric,
        topk::Integer,
        recommender_type::Type{<:Recommender},
        data::DataAccessor,
        recommender_args...
    )

Conduct `n_folds` cross validation for a combination of recommender `recommender_type` and ranking metric `metric`. A recommender is initialized with `recommender_args` and runs top-`k` recommendation.
"""
function cross_validation(n_folds::Integer, metric::RankingMetric, topk::Integer, recommender_type::Type{<:Recommender}, data::DataAccessor, recommender_args...)
    cross_validation(n_folds, [metric], topk, recommender_type, data, recommender_args...)[1]
end

function cross_validation(n_folds::Integer, metrics::AbstractVector{T}, topk::Integer,
                          recommender_type::Type{<:Recommender}, data::DataAccessor, recommender_args...
                         ) where T<:RankingMetric
    accum_accuracy = zeros(length(metrics))
    for (train_data, truth_data) in split_data(data, n_folds)
        recommender = recommender_type(train_data, recommender_args...)
        fit!(recommender)
        accum_accuracy += evaluate(recommender, truth_data, metrics, topk)
    end
    accum_accuracy / n_folds
end

"""
    cross_validation(
        n_folds::Integer,
        metric::AccuracyMetric,
        recommender_type::Type{<:Recommender},
        data::DataAccessor,
        recommender_args...
    )

Conduct `n_folds` cross validation for a combination of recommender `recommender_type` and accuracy metric `metric`. A recommender is initialized with `recommender_args`.
"""
function cross_validation(n_folds::Integer, metric::AccuracyMetric, recommender_type::Type{<:Recommender}, data::DataAccessor, recommender_args...)
    cross_validation(n_folds, [metric], recommender_type, data, recommender_args...)[1]
end

function cross_validation(n_folds::Integer, metrics::AbstractVector{T},
                          recommender_type::Type{<:Recommender}, data::DataAccessor, recommender_args...
                         ) where T<:AccuracyMetric
    accum_accuracy = zeros(length(metrics))
    for (train_data, truth_data) in split_data(data, n_folds)
        recommender = recommender_type(train_data, recommender_args...)
        fit!(recommender)
        accum_accuracy = evaluate(recommender, truth_data, metrics)
    end
    accum_accuracy / n_folds
end

"""
    leave_one_out(
        metric::RankingMetric,
        topk::Integer,
        recommender_type::Type{<:Recommender},
        data::DataAccessor,
        recommender_args...
    )

Conduct leave-one-out cross validation (LOOCV) for a combination of recommender `recommender_type` and accuracy metric `metric`. A recommender is initialized with `recommender_args` and runs top-`k` recommendation.
"""
function leave_one_out(metric::RankingMetric, topk::Integer, recommender_type::Type{<:Recommender}, data::DataAccessor, recommender_args...)
    cross_validation(length(data.events), metric, topk, recommender_type, data, recommender_args...)
end

"""
    leave_one_out(
        metric::AccuracyMetric,
        recommender_type::Type{<:Recommender},
        data::DataAccessor,
        recommender_args...
    )

Conduct leave-one-out cross validation (LOOCV) for a combination of recommender `recommender_type` and accuracy metric `metric`. A recommender is initialized with `recommender_args`.
"""
function leave_one_out(metric::AccuracyMetric, recommender_type::Type{<:Recommender}, data::DataAccessor, recommender_args...)
    cross_validation(length(data.events), metric, recommender_type, data, recommender_args...)
end

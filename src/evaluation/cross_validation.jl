export cross_validation, leave_one_out

"""
    cross_validation(
        n_folds::Integer,
        metric::Type{<:RankingMetric},
        topk::Integer,
        recommender_type::Type{<:Recommender},
        data::DataAccessor,
        recommender_args...
    )

Conduct `n_folds` cross validation for a combination of recommender `recommender_type` and ranking metric `metric`. A recommender is initialized with `recommender_args` and runs top-`k` recommendation.
"""
function cross_validation(n_folds::Integer, metric::Type{<:RankingMetric}, topk::Integer, recommender_type::Type{<:Recommender}, data::DataAccessor, recommender_args...)
    accum_accuracy = 0.0
    for (train_data, truth_data) in split_events(data, n_folds)
        recommender = recommender_type(train_data, recommender_args...)
        fit!(recommender)
        accum_accuracy += evaluate(recommender, truth_data, metric(), topk)
    end
    accum_accuracy / n_folds
end

"""
    cross_validation(
        n_folds::Integer,
        metric::Type{<:AccuracyMetric},
        recommender_type::Type{<:Recommender},
        data::DataAccessor,
        recommender_args...
    )

Conduct `n_folds` cross validation for a combination of recommender `recommender_type` and accuracy metric `metric`. A recommender is initialized with `recommender_args`.
"""
function cross_validation(n_folds::Integer, metric::Type{<:AccuracyMetric}, recommender_type::Type{<:Recommender}, data::DataAccessor, recommender_args...)
    accum_accuracy = 0.0
    for (train_data, truth_data) in split_events(data, n_folds)
        recommender = recommender_type(train_data, recommender_args...)
        fit!(recommender)
        accum_accuracy = evaluate(recommender, truth_data, metric())
    end
    accum_accuracy / n_folds
end

"""
    leave_one_out(
        metric::Type{<:RankingMetric},
        topk::Integer,
        recommender_type::Type{<:Recommender},
        data::DataAccessor,
        recommender_args...
    )

Conduct leave-one-out cross validation (LOOCV) for a combination of recommender `recommender_type` and accuracy metric `metric`. A recommender is initialized with `recommender_args` and runs top-`k` recommendation.
"""
function leave_one_out(metric::Type{<:RankingMetric}, topk::Integer, recommender_type::Type{<:Recommender}, data::DataAccessor, recommender_args...)
    cross_validation(length(data.events), metric, topk, recommender_type, data, recommender_args...)
end

"""
    leave_one_out(
        metric::Type{<:AccuracyMetric},
        recommender_type::Type{<:Recommender},
        data::DataAccessor,
        recommender_args...
    )

Conduct leave-one-out cross validation (LOOCV) for a combination of recommender `recommender_type` and accuracy metric `metric`. A recommender is initialized with `recommender_args`.
"""
function leave_one_out(metric::Type{<:AccuracyMetric}, recommender_type::Type{<:Recommender}, data::DataAccessor, recommender_args...)
    cross_validation(length(data.events), metric, recommender_type, data, recommender_args...)
end

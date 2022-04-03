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
    accum = 0.0
    for (train_data, truth_data) in split_events(data, n_folds)
        # get recommender from the specified data type
        recommender = recommender_type(train_data, recommender_args...)
        fit!(recommender)
        accuracy = evaluate(recommender, truth_data, metric(), topk)
        if isnan(accuracy)
            @warn "cannot calculate a metric for $truth_data"
            continue
        end
        accum += accuracy
    end
    accum / n_folds
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
    accum = 0.0
    for (train_data, truth_data) in split_events(data, n_folds)
        # get recommender from the specified data type
        recommender = recommender_type(train_data, recommender_args...)
        fit!(recommender)
        accuracy = evaluate(recommender, truth_data, metric())
        if isnan(accuracy)
            @warn "cannot calculate a metric for $truth_data"
            continue
        end
        accum += accuracy
    end
    accum / n_folds
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

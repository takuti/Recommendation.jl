export cross_validation

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

    if n_folds < 2
        error("`n_folds` must be greater than 1 to split the samples into train and test sets.")
    end

    events = shuffle(data.events)
    n_events = length(events)

    if n_folds > n_events
        error("`n_folds = $n_folds` must be less than $n_events, the number of all samples.")
    end

    n_users, n_items = size(data.R)

    step = convert(Integer, round(n_events / n_folds))
    accum = 0.0

    if n_folds == n_events
        @info "Leave-one-out cross validation against $n_events samples"
    else
        @info "$n_folds-fold cross validation against $n_events samples"
    end

    for (index, head) in enumerate(1:step:n_events)
        tail = min(head + step - 1, n_events)

        truth_events = events[head:tail]
        truth_data = DataAccessor(truth_events, n_users, n_items)

        train_events = vcat(events[1:head - 1], events[tail + 1:end])
        train_data = DataAccessor(train_events, n_users, n_items)

        # get recommender from the specified data type
        recommender = recommender_type(train_data, recommender_args...)
        fit!(recommender)

        accuracy = evaluate(recommender, truth_data, metric(), topk)
        @info "fold#$index: tested with samples in [$head, $tail]. $metric = $accuracy"
        if isnan(accuracy); continue; end
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

    if n_folds < 2
        error("`n_folds` must be greater than 1 to split the samples into train and test sets.")
    end

    events = shuffle(data.events)
    n_events = length(events)

    if n_folds > n_events
        error("`n_folds = $n_folds` must be less than $n_events, the number of all samples.")
    end

    n_users, n_items = size(data.R)

    step = convert(Integer, round(n_events / n_folds))
    accum = 0.0

    if n_folds == n_events
        @info "Leave-one-out cross validation against $n_events samples"
    else
        @info "$n_folds-fold cross validation against $n_events samples"
    end

    for (index, head) in enumerate(1:step:n_events)
        tail = min(head + step - 1, n_events)

        truth_events = events[head:tail]
        truth_data = DataAccessor(truth_events, n_users, n_items)

        train_events = vcat(events[1:head - 1], events[tail + 1:end])
        train_data = DataAccessor(train_events, n_users, n_items)

        # get recommender from the specified data type
        recommender = recommender_type(train_data, recommender_args...)
        fit!(recommender)

        accuracy = evaluate(recommender, truth_data, metric())
        @info "fold#$index: tested with samples in [$head, $tail]. $metric = $accuracy"
        if isnan(accuracy); continue; end
        accum += accuracy
    end

    accum / n_folds
end

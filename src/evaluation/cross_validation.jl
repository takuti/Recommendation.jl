export cross_validation

"""
    cross_validation(
        n_fold::Int,
        metric::Type{<:Metric},
        k::Int,
        recommender_type::Type{<:Recommender},
        data::DataAccessor,
        recommender_args...
    )

Conduct `n_fold` cross validation for a combination of recommender `recommender_type` and metric `metric`. A recommender is initialized with `recommender_args`. For ranking metric, accuracy is measured by top-`k` recommendation.
"""
function cross_validation(n_fold::Int, metric::Type{<:Metric}, k::Int, recommender_type::Type{<:Recommender}, data::DataAccessor, recommender_args...)

    n_user, n_item = size(data.R)

    events = shuffle(data.events)
    n_events = length(events)

    step = convert(Int, round(n_events / n_fold))
    accum = 0.0

    for head in 1:step:n_events
        tail = min(head + step - 1, n_events)

        truth_events = events[head:tail]
        truth_data = DataAccessor(truth_events, n_user, n_item)

        train_events = vcat(events[1:head - 1], events[tail + 1:end])
        train_data = DataAccessor(train_events, n_user, n_item)

        # get recommender from the specified data type
        recommender = recommender_type(train_data, recommender_args...)
        build!(recommender)

        accum += evaluate(recommender, truth_data, metric(), k)
    end

    accum / n_fold
end

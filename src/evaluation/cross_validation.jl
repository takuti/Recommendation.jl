export cross_validation

"""
    cross_validation(
        n_fold::Int,
        metric::Type{S<:Metric},
        k::Int,
        rec_type::Type{T<:Recommender},
        da::DataAccessor,
        rec_args...
    )

Conduct `n_fold` cross validation for a combination of recommender `rec_type` and metric `metric`. A recommender is initialized with `rec_args`. For ranking metric, accuracy is measured by top-`k` recommendation.
"""
function cross_validation(n_fold::Int, metric::Type{S}, k::Int, rec_type::Type{T}, da::DataAccessor, rec_args...) where {T<:Recommender,S<:Metric}

    n_user, n_item = size(da.R)

    events = shuffle(da.events)
    n_events = length(events)

    step = convert(Int, round(n_events / n_fold))
    accum = 0.0

    for head in 1:step:n_events
        tail = min(head + step - 1, n_events)

        truth_events = events[head:tail]
        truth_da = DataAccessor(truth_events, n_user, n_item)

        train_events = vcat(events[1:head - 1], events[tail + 1:end])
        train_da = DataAccessor(train_events, n_user, n_item)

        # get recommender from the specified data type
        rec = rec_type(train_da, rec_args...)
        build(rec)

        accum += evaluate(rec, truth_da, metric(), k)
    end

    accum / n_fold
end

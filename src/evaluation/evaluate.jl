export evaluate, check_metrics_type

function evaluate(recommender::Recommender, truth_data::DataAccessor,
                  metric::AccuracyMetric)
    evaluate(recommender, truth_data, [metric])[1]
end

function evaluate(recommender::Recommender, truth_data::DataAccessor,
                  metrics::AbstractVector{T}) where {T<:AccuracyMetric}
    validate(recommender, truth_data)

    nonzero_indices = findall(!iszero, truth_data.R)
    truth = truth_data.R[nonzero_indices]
    pred = predict(recommender, nonzero_indices)
    [measure(metric, truth, pred) for metric in metrics]
end

function evaluate(recommender::Recommender, truth_data::DataAccessor,
                  metric::Metric, topk::Integer; allow_repeat=false)
    evaluate(recommender, truth_data, [metric], topk, allow_repeat=allow_repeat)[1]
end

function check_metrics_type(metrics::AbstractVector{T},
                            accepted_type::Type{<:Metric}) where {T<:Metric}
    if !all(metric -> typeof(metric) <: accepted_type, metrics)
        error("$metrics contains something different from $accepted_type")
    end
end

function evaluate(recommender::Recommender, truth_data::DataAccessor,
                  metrics::AbstractVector{T}, topk::Integer; allow_repeat=false) where {T<:Metric}
    validate(recommender, truth_data)
    check_metrics_type(metrics, Union{RankingMetric, AggregatedMetric, Coverage, Novelty})

    n_users, n_items = size(truth_data.R)
    all_items = 1:n_items

    accums = [Threads.Atomic{Float64}(0.0) for i in 1:length(metrics)]
    recommendations = Vector{Vector{Integer}}()
    recs_lock = Threads.ReentrantLock()

    Threads.@threads for u in 1:n_users
        observed_items = findall(!iszero, truth_data.R[u, :])
        nnz = length(observed_items)
        if nnz == 0
            continue
        end
        if allow_repeat
            candidates = all_items
        else
            # items that were unobserved as of building the model
            candidates = findall(iszero, recommender.data.R[u, :])
        end
        pred = [first(item_score_pair) for item_score_pair in recommend(recommender, u, topk, candidates)]

        truth = partialsortperm(truth_data.R[u, :], 1:nnz, rev=true)
        for (i, metric) in enumerate(metrics)
            if typeof(metric) <: RankingMetric
                Threads.atomic_add!(accums[i], measure(metric, truth, pred, topk))
            elseif typeof(metric) <: Coverage
                Threads.atomic_add!(accums[i], measure(metric, pred, catalog=all_items))
            elseif typeof(metric) <: Novelty
                Threads.atomic_add!(accums[i], float(measure(metric, pred, observed=observed_items)))
            end
        end
        lock(recs_lock) do
            push!(recommendations, pred)
        end
    end

    # return average accuracy over the all target users
    res = [accum[] for accum in accums] ./ n_users
    for (i, metric) in enumerate(metrics)
        if typeof(metric) <: AggregatedMetric
            res[i] = measure(metric, recommendations, topk=topk)
        end
    end
    res
end

export evaluate, check_metrics_type

function evaluate(recommender::Recommender, truth_data::DataAccessor,
                  metric::AccuracyMetric)
    evaluate(recommender, truth_data, [metric])[1]
end

function evaluate(recommender::Recommender, truth_data::DataAccessor,
                  metrics::AbstractVector{T}) where {T<:AccuracyMetric}
    validate(recommender, truth_data)

    nonzero_indices = findall(!iszero, truth_data.R)

    truth = zeros(length(nonzero_indices))
    pred = zeros(length(nonzero_indices))
    for (j, idx) in enumerate(nonzero_indices)
        truth[j] = truth_data.R[idx]
        pred[j] = predict(recommender, idx[1], idx[2])
    end

    [measure(metric, truth, pred) for metric in metrics]
end

function evaluate(recommender::Recommender, truth_data::DataAccessor,
                  metric::Metric, topk::Integer)
    evaluate(recommender, truth_data, [metric], topk)[1]
end

function check_metrics_type(metrics::AbstractVector{Metric}, accepted_type::Type{<:Metric})
    if !all(metric -> typeof(metric) <: accepted_type, metrics)
        error("$metrics contains something different from $accepted_type")
    end
end

function evaluate(recommender::Recommender, truth_data::DataAccessor,
                  metrics::AbstractVector{Metric}, topk::Integer)
    validate(recommender, truth_data)
    check_metrics_type(metrics, Union{RankingMetric, AggregatedMetric})

    n_users, n_items = size(truth_data.R)

    accums = [Threads.Atomic{Float64}(0.0) for i in 1:length(metrics)]
    recommendations = Vector{Vector{Integer}}()

    Threads.@threads for u in 1:n_users
        nnz = sum(!iszero, truth_data.R[u, :])
        if nnz == 0
            continue
        end
        truth = partialsortperm(truth_data.R[u, :], 1:nnz, rev=true)
        candidates = findall(iszero, recommender.data.R[u, :]) # items that were unobserved as of building the model
        pred = [first(item_score_pair) for item_score_pair in recommend(recommender, u, topk, candidates)]
        for (i, metric) in enumerate(metrics)
            if typeof(metric) <: RankingMetric
                Threads.atomic_add!(accums[i], measure(metric, truth, pred, topk))
            end
        end
        push!(recommendations, pred)
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

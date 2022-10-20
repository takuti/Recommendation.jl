export evaluate

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
                  metric::RankingMetric, topk::Integer)
    evaluate(recommender, truth_data, [metric], topk)[1]
end

function evaluate(recommender::Recommender, truth_data::DataAccessor,
                  metrics::AbstractVector{T}, topk::Integer) where {T<:RankingMetric}
    validate(recommender, truth_data)
    n_users, n_items = size(truth_data.R)

    accums = [Threads.Atomic{Float64}(0.0) for i in 1:length(metrics)]

    Threads.@threads for u in 1:n_users
        truth = filter(idx -> !iszero(truth_data.R[u, idx]), sortperm(truth_data.R[u, :], rev=true))
        if length(truth) == 0
            continue
        end
        candidates = findall(iszero, recommender.data.R[u, :]) # items that were unobserved as of building the model
        pred = [first(item_score_pair) for item_score_pair in recommend(recommender, u, topk, candidates)]
        for (i, metric) in enumerate(metrics)
            Threads.atomic_add!(accums[i], measure(metric, truth, pred, topk))
        end
    end

    # return average accuracy over the all target users
    [accum[] for accum in accums] ./ n_users
end

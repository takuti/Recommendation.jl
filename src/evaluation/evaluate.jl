export evaluate

function evaluate(recommender::Recommender, truth_data::DataAccessor,
                  metric::AccuracyMetric)
    validate(recommender, truth_data)

    nonzero_indices = findall(!iszero, truth_data.R)

    truth = zeros(length(nonzero_indices))
    pred = zeros(length(nonzero_indices))
    for (j, idx) in enumerate(nonzero_indices)
        truth[j] = truth_data.R[idx]
        pred[j] = predict(recommender, idx[1], idx[2])
    end

    measure(metric, truth, pred)
end

function evaluate(recommender::Recommender, truth_data::DataAccessor,
                  metric::RankingMetric, topk::Integer)
    validate(recommender, truth_data)
    n_users, n_items = size(truth_data.R)

    accum = 0.0

    candidates = Array(1:n_items)
    for u in 1:n_users
        truth = [first(t) for t in sort(collect(zip(candidates, truth_data.R[u, :])), by=t->last(t), rev=true)]
        recos = recommend(recommender, u, topk, candidates)
        pred = [first(t) for t in sort(recos, by=t->last(t), rev=true)]
        accum += measure(metric, truth, pred, topk)
    end

    # return average accuracy over the all target users
    accum / n_users
end

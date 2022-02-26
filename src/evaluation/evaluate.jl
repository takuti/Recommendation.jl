export evaluate

function evaluate(recommender::Recommender, truth_data::DataAccessor,
                  metric::AccuracyMetric)
    validate(recommender, truth_data)
    n_users, n_items = size(truth_data.R)

    accum = 0.0

    for u in 1:n_users
        pred = zeros(n_items)
        for i in 1:n_items
            pred[i] = predict(recommender, u, i)
        end
        accum += measure(metric, truth_data.R[u, :], pred)
    end

    # return average accuracy over the all target users
    accum / n_users
end

function evaluate(recommender::Recommender, truth_data::DataAccessor,
                  metric::RankingMetric, k::Integer=0)
    validate(recommender, truth_data)
    n_users, n_items = size(truth_data.R)

    accum = 0.0

    candidates = Array(1:n_items)
    for u in 1:n_users
        truth = [first(t) for t in sort(collect(zip(candidates, truth_data.R[u, :])), by=t->last(t), rev=true)]
        recos = recommend(recommender, u, k, candidates)
        pred = [first(t) for t in sort(recos, by=t->last(t), rev=true)]
        accum += measure(metric, truth, pred, k)
    end

    # return average accuracy over the all target users
    accum / n_users
end

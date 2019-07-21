export evaluate

function validate_size(recommender::Recommender, truth_data::DataAccessor)
    n_rec_user, n_rec_item = size(recommender.data.R)
    n_truth_user, n_truth_item = size(truth_data.R)

    if n_rec_user != n_truth_user
        error("number of users is mismatched: (recommenre, truth) = ($(n_rec_user), $(n_truth_user)")
    elseif n_rec_item != n_truth_item
        error("number of items is mismatched: (recommenre, truth) = ($(n_rec_item), $(n_truth_item)")
    end

    n_truth_user, n_truth_item
end

function evaluate(recommender::Recommender, truth_data::DataAccessor,
                  metric::AccuracyMetric, k::Int=0)
    check_build_status(recommender)
    n_user, n_item = validate_size(recommender, truth_data)

    accum = 0.0

    for u in 1:n_user
        pred = zeros(n_item)
        for i in 1:n_item
            pred[i] = predict(recommender, u, i)
        end
        accum += measure(metric, truth_data.R[u, :], pred)
    end

    # return average accuracy over the all target users
    accum / n_user
end

function evaluate(recommender::Recommender, truth_data::DataAccessor,
                  metric::RankingMetric, k::Int=0)
    check_build_status(recommender)
    n_user, n_item = validate_size(recommender, truth_data)

    accum = 0.0

    candidates = Array(1:n_item)
    for u in 1:n_user
        truth = [first(t) for t in sort(collect(zip(candidates, truth_data.R[u, :])), by=t->last(t), rev=true)]
        recos = recommend(recommender, u, k, candidates)
        pred = [first(t) for t in sort(recos, by=t->last(t), rev=true)]
        accum += measure(metric, truth, pred, k)
    end

    # return average accuracy over the all target users
    accum / n_user
end

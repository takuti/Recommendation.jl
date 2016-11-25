export evaluate

function evaluate(rec::Recommender, truth_da::DataAccessor,
                  metric::AccuracyMetric)

    # if the recommender has not been built yet, build here
    if !haskey(rec.states, :is_built) || !rec.states[:is_built]
        build(rec)
    end

    n_rec_user, n_rec_item = size(rec.da.R)
    n_truth_user, n_truth_item = size(truth_da.R)

    if n_rec_user != n_truth_user
        error("number of users is mismatched: (recommenre, truth) = ($(n_rec_user), $(n_truth_user)")
    elseif n_rec_item != n_truth_item
        error("number of items is mismatched: (recommenre, truth) = ($(n_rec_item), $(n_truth_item)")
    end

    accum = 0.0

    for u in 1:n_truth_user
        pred = zeros(n_truth_item)
        for i in 1:n_truth_item
            pred[i] = predict(rec, u, i)
        end
        accum += measure(metric, truth_da.R[u, :], pred)
    end

    # return average RMSE over the all target users
    accum / n_truth_user
end

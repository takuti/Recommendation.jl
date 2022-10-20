using Recommendation

recommenders = [
    # (recommender => params, accuracy_metrics, ranking_metrics)
    (ItemMean => [], true, true),
    (MostPopular => [], false, true),
    (ThresholdPercentage => [3.0], false, true),
    (UserMean => [], true, true),
    # CoOccurrence => [1],
    # BPRMatrixFactorization => [],
    # FactorizationMachines => [],
    # ItemKNN => [5],
    # MatrixFactorization => [],
    # SVD => [],
    # TfIdf => [],
    # UserKNN => [5, true]
]

accuracy_metrics = [
    RMSE,
    MAE,
]

ranking_metrics = [
    Recall,
    Precision,
    AUC,
    ReciprocalRank,
    MPR,
    NDCG
]

# metrics = [
#     AggregatedDiversity,
#     ShannonEntropy,
#     GiniIndex,
#     Coverage,
#     Novelty,
#     IntraListSimilarity,
#     Serendipity,
# ]

datasets = [
    load_movielens_100k,
    # load_movielens_latest,
    # load_amazon_review,
    # load_lastfm
]

test_ratio = 0.2
topk = 10

for dataset in datasets
    @info "Dataset: $dataset"
    data = dataset()
    train_data, truth_data = split_data(data, test_ratio)
    for ((recommender, params), use_accuracy_metrics, use_ranking_metrics) in recommenders
        @info "Recommender: $recommender"
        r = recommender(train_data, params...)
        fit!(r)

        # accuracy metrics
        if use_accuracy_metrics
            results = evaluate(r, truth_data, [metric() for metric in accuracy_metrics])
            for (metric, res) in zip(accuracy_metrics, results)
                @info "$metric = $res"
            end
        end

        # ranking metrics
        if use_ranking_metrics
            results = evaluate(r, truth_data, [metric() for metric in ranking_metrics], topk)
            for (metric, res) in zip(ranking_metrics, results)
                @info "$metric = $res"
            end
        end
    end
end

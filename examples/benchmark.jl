using Recommendation

recommenders = Dict(
    # CoOccurrence => [1],
    ItemMean => [],
    # MostPopular => [],
    # ThresholdPercentage => [3.0],
    UserMean => [],
    # BPRMatrixFactorization => [],
    # FactorizationMachines => [],
    # ItemKNN => [5],
    # MatrixFactorization => [],
    # SVD => [],
    # TfIdf => [],
    # UserKNN => [5, true]
)

metrics = [
    RMSE,
    MAE,
    # AggregatedDiversity,
    # ShannonEntropy,
    # GiniIndex,
    # Coverage,
    # Novelty,
    # IntraListSimilarity,
    # Serendipity,
    # Recall,
    # Precision,
    # AUC,
    # ReciprocalRank,
    # MPR,
    # NDCG
]

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
    for (recommender, params) in recommenders
        @info "Recommender: $recommender"
        r = recommender(train_data, params...)
        fit!(r)

        # accuracy metrics
        results = evaluate(r, truth_data, [metric() for metric in accuracy_metrics])
        for (metric, res) in zip(accuracy_metrics, results)
            @info "$metric = $res"
        end

        # ranking metrics
        results = evaluate(r, truth_data, [metric() for metric in ranking_metrics], topk)
        for (metric, res) in zip(ranking_metrics, results)
            @info "$metric = $res"
        end
    end
end

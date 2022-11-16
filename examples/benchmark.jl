# If this script is executed directly from the Recommendation.jl repository,
# instantiate the package from the local code so everything is up-to-date.
# Otherwise, we assume Recommendation.jl is pre-installed, and
# `using Recommendation` works without manual activation.
proj_toml_path = normpath(joinpath(@__DIR__, "..", "Project.toml"))
if isfile(proj_toml_path)
    using TOML
    conf = TOML.parse(read(proj_toml_path, String))
    if conf["name"] == "Recommendation"
        using Pkg
        Pkg.activate(dirname(proj_toml_path))
        Pkg.instantiate()
    end
end

using Recommendation

# recommenders that predict values (e.g., ratings) for missing user-item pairs,
# which enable the outputs to be evaluated by accuracy metrics.
value_prediction_recommenders = [
    ItemMean => [],
    UserMean => [],
    SVD => [16],
    SVD => [32],
    SVD => [64],
    # BPRMatrixFactorization => [],
    # FactorizationMachines => [],
    # MatrixFactorization => [],
    # ItemKNN => [5],
    # UserKNN => [5, true],
]

# recommenders that order missing user-item pairs by their own definition of
# "scores" such as popularity count and item co-occurrence. They cannot be
# evaluated by accuracy measures, since the resulting scores are not directly
# approximating the original user-item data.
rank_by_score_recommenders = [
    MostPopular => [],
    ThresholdPercentage => [3.0],
    # CoOccurrence => [1],
    # TfIdf => [],
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
    NDCG,
    AggregatedDiversity,
    ShannonEntropy,
    GiniIndex,
]

# metrics = [
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

    @info "Evaluating value prediction-based recommenders"
    for (recommender, params) in value_prediction_recommenders
        @info "Recommender: $recommender($params...)"
        r = recommender(train_data, params...)
        fit!(r)

        # accuracy metrics
        results = evaluate(r, truth_data, [metric() for metric in accuracy_metrics])
        for (metric, res) in zip(accuracy_metrics, results)
            @info "  $metric = $res"
        end

        # ranking / aggregated metrics
        results = evaluate(r, truth_data, [metric() for metric in ranking_metrics], topk)
        for (metric, res) in zip(ranking_metrics, results)
            @info "  $metric = $res"
        end
    end

    @info "Evaluating custom ranking score-based recommenders"
    for (recommender, params) in rank_by_score_recommenders
        @info "Recommender: $recommender($params...)"
        r = recommender(train_data, params...)
        fit!(r)

        # ranking / aggregated metrics
        results = evaluate(r, truth_data, [metric() for metric in ranking_metrics], topk)
        for (metric, res) in zip(ranking_metrics, results)
            @info "  $metric = $res"
        end
    end
end

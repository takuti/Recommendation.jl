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
    SVD => [4],
    SVD => [8],
    SVD => [16],
    SVD => [32],
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
    # ThresholdPercentage => [3.0],
    # CoOccurrence => [1],
    # TfIdf => [],
]

function instantiate(metrics::AbstractVector{DataType})
    [metric() for metric in metrics]
end

accuracy_metrics = instantiate([
    RMSE,
    MAE,
])

topk_metrics = instantiate([
    Recall,
    Precision,
    AUC,
    ReciprocalRank,
    MPR,
    NDCG,
    AggregatedDiversity,
    ShannonEntropy,
    GiniIndex,
])

# * IntraListSimilarity can be calculated with an item-item similarity metrix, which can be built by ItemKNN.
# * Serendipity requires context-specific definition of relevance and unexpectedness.
intra_list_metrics = instantiate([
    Coverage,
    Novelty
])

datasets = [
    load_movielens_100k,
    # load_movielens_latest,
    # load_amazon_review,
    # load_lastfm
]

topk = 10
n_folds = 5

for dataset in datasets
    @info "Dataset: $dataset"
    data = dataset()

    @info "Evaluating value prediction-based recommenders"
    for (recommender, params) in value_prediction_recommenders
        @info "Recommender: $recommender($params...)"
        results = cross_validation(n_folds, accuracy_metrics, recommender, data, params...)
        for (metric, res) in zip(accuracy_metrics, results)
            @info "  $metric = $res"
        end
        results = cross_validation(n_folds, topk_metrics, topk, recommender, data, params...)
        for (metric, res) in zip(topk_metrics, results)
            @info "  $metric = $res"
        end
        results = cross_validation(n_folds, intra_list_metrics, topk, recommender, data, params..., allow_repeat=true)
        for (metric, res) in zip(intra_list_metrics, results)
            @info "  $metric = $res"
        end
    end

    @info "Evaluating custom ranking score-based recommenders"
    for (recommender, params) in rank_by_score_recommenders
        @info "Recommender: $recommender($params...)"
        results = cross_validation(n_folds, topk_metrics, topk, recommender, data, params...)
        for (metric, res) in zip(topk_metrics, results)
            @info "  $metric = $res"
        end
        results = cross_validation(n_folds, intra_list_metrics, topk, recommender, data, params..., allow_repeat=true)
        for (metric, res) in zip(intra_list_metrics, results)
            @info "  $metric = $res"
        end
    end
end

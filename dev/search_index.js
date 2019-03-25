var documenterSearchIndex = {"docs": [

{
    "location": "#",
    "page": "Home",
    "title": "Home",
    "category": "page",
    "text": ""
},

{
    "location": "#Recommendation.jl-1",
    "page": "Home",
    "title": "Recommendation.jl",
    "category": "section",
    "text": "Recommendation.jl is a Julia package for building recommender systems. Thanks to independent data accessor and recommender implementations, this package enables you to build recommendation systems on your own data and algorithms.Pages = [\n    \"baseline.md\",\n    \"cf.md\",\n    \"content_based.md\",\n    \"evaluation.md\",\n]\nDepth = 1For more information, you can refer to my article."
},

{
    "location": "getting_started/#",
    "page": "Getting Started",
    "title": "Getting Started",
    "category": "page",
    "text": ""
},

{
    "location": "getting_started/#Getting-Started-1",
    "page": "Getting Started",
    "title": "Getting Started",
    "category": "section",
    "text": ""
},

{
    "location": "getting_started/#Installation-1",
    "page": "Getting Started",
    "title": "Installation",
    "category": "section",
    "text": "This package is registered in METADATA.jl.Pkg.add(\"Recommendation\")"
},

{
    "location": "getting_started/#Usage-1",
    "page": "Getting Started",
    "title": "Usage",
    "category": "section",
    "text": "This package contains DataAccessor and several fundamental recommendation techniques (e.g., non-personalized MostPopular recommender, CF and MF), and evaluation metrics such as Recall. All of them can be accessible by loading the package as follows:using RecommendationFirst of all, you need to create a data accessor from a matrix:using SparseArrays\n\nda = DataAccessor(sparse([1 0 0; 4 5 0]))or set of events:const n_user = 5\nconst n_item = 10\n\nevents = [Event(1, 2, 1), Event(3, 2, 1), Event(2, 6, 4)]\n\nda = DataAccessor(events, n_user, n_item)where Event() is a composite type which represents a user-item interaction:type Event\n    user::Int\n    item::Int\n    value::Float64\nendNext, you can pass the data accessor to an arbitrary recommender as:recommender = MostPopular(da)and building a recommendation engine should be easy:build(recommender)Personalized recommenders sometimes require us to specify the hyperparameters:recommender = MF(da, Parameters(:k => 2))\nbuild(recommender, learning_rate=15e-4, max_iter=100)Once a recommendation engine has been built successfully, top-k recommendation for a user u with item candidates candidates is performed as follows:u = 4\nk = 2\ncandidates = [i for i in 1:n_item] # all items\n\nrecommend(recommender, u, k, candidates)"
},

{
    "location": "baseline/#",
    "page": "Non-Personalized Baselines",
    "title": "Non-Personalized Baselines",
    "category": "page",
    "text": ""
},

{
    "location": "baseline/#Recommendation.CoOccurrence",
    "page": "Non-Personalized Baselines",
    "title": "Recommendation.CoOccurrence",
    "category": "type",
    "text": "CoOccurrence(\n    da::DataAccessor,\n    hyperparams::Parameters=Parameters(:i_ref => 1)\n)\n\nRecommend items which are most frequently co-occurred with a reference item i_ref.\n\n\n\n\n\n"
},

{
    "location": "baseline/#Recommendation.MostPopular",
    "page": "Non-Personalized Baselines",
    "title": "Recommendation.MostPopular",
    "category": "type",
    "text": "MostPopular(da::DataAccessor)\n\nRecommend most popular items.\n\n\n\n\n\n"
},

{
    "location": "baseline/#Recommendation.ThresholdPercentage",
    "page": "Non-Personalized Baselines",
    "title": "Recommendation.ThresholdPercentage",
    "category": "type",
    "text": "ThresholdPercentage(\n    da::DataAccessor,\n    hyperparams::Parameters=Parameters(:th => 2.5)\n)\n\nRecommend based on percentage of ratings which are greater than a certain threshold value th.\n\n\n\n\n\n"
},

{
    "location": "baseline/#Recommendation.UserMean",
    "page": "Non-Personalized Baselines",
    "title": "Recommendation.UserMean",
    "category": "type",
    "text": "UserMean(da::DataAccessor)\n\nRecommend based on global user mean rating.\n\n\n\n\n\n"
},

{
    "location": "baseline/#Recommendation.ItemMean",
    "page": "Non-Personalized Baselines",
    "title": "Recommendation.ItemMean",
    "category": "type",
    "text": "ItemMean(da::DataAccessor)\n\nRecommend based on global item mean rating.\n\n\n\n\n\n"
},

{
    "location": "baseline/#Non-Personalized-Baselines-1",
    "page": "Non-Personalized Baselines",
    "title": "Non-Personalized Baselines",
    "category": "section",
    "text": "Pages = [\"baseline.md\"]CoOccurrence\nMostPopular\nThresholdPercentage\nUserMean\nItemMean"
},

{
    "location": "cf/#",
    "page": "Collaborative Filtering",
    "title": "Collaborative Filtering",
    "category": "page",
    "text": ""
},

{
    "location": "cf/#Recommendation.ItemKNN",
    "page": "Collaborative Filtering",
    "title": "Recommendation.ItemKNN",
    "category": "type",
    "text": "ItemKNN(\n    da::DataAccessor,\n    hyperparams::Parameters=Parameters(:k => 5)\n)\n\nItem-based collaborative filtering. k represents number of neighbors.\n\n\n\n\n\n"
},

{
    "location": "cf/#Recommendation.UserKNN",
    "page": "Collaborative Filtering",
    "title": "Recommendation.UserKNN",
    "category": "type",
    "text": "UserKNN(\n    da::DataAccessor,\n    hyperparams::Parameters=Parameters(:k => 5),\n    is_normalized::Bool=false\n)\n\nUser-based collaborative filtering. k represents number of neighbors, and is_normalized specifies if weighted sum of neighbors\' rating is normalized.\n\n\n\n\n\n"
},

{
    "location": "cf/#Recommendation.SVD",
    "page": "Collaborative Filtering",
    "title": "Recommendation.SVD",
    "category": "type",
    "text": "SVD(\n    da::DataAccessor,\n    hyperparams::Parameters=Parameters(:k => 20)\n)\n\nRecommendation based on Singular Value Decomposition (SVD). Number of factors is configured by k.\n\n\n\n\n\n"
},

{
    "location": "cf/#Recommendation.MF",
    "page": "Collaborative Filtering",
    "title": "Recommendation.MF",
    "category": "type",
    "text": "MF(\n    da::DataAccessor,\n    hyperparams::Parameters=Parameters(:k => 20)\n)\n\nRecommendation based on Matrix Factorization (MF). Number of factors is configured by k.\n\n\n\n\n\n"
},

{
    "location": "cf/#Collaborative-Filtering-1",
    "page": "Collaborative Filtering",
    "title": "Collaborative Filtering",
    "category": "section",
    "text": "Pages = [\"cf.md\"]ItemKNN\nUserKNN\nSVD\nMF"
},

{
    "location": "content_based/#",
    "page": "Content-Based Recommenders",
    "title": "Content-Based Recommenders",
    "category": "page",
    "text": ""
},

{
    "location": "content_based/#Recommendation.TFIDF",
    "page": "Content-Based Recommenders",
    "title": "Recommendation.TFIDF",
    "category": "type",
    "text": "TFIDF(\n    da::DataAccessor,\n    tf::AbstractMatrix,\n    idf::AbstractMatrix\n)\n\nContent-based recommendation using TF-IDF scoring. TF and IDF matrix are respectively specified as tf and idf.\n\n\n\n\n\n"
},

{
    "location": "content_based/#Content-Based-Recommenders-1",
    "page": "Content-Based Recommenders",
    "title": "Content-Based Recommenders",
    "category": "section",
    "text": "Pages = [\"content_based.md\"]TFIDF"
},

{
    "location": "evaluation/#",
    "page": "Evaluation",
    "title": "Evaluation",
    "category": "page",
    "text": ""
},

{
    "location": "evaluation/#Evaluation-1",
    "page": "Evaluation",
    "title": "Evaluation",
    "category": "section",
    "text": "Pages = [\"evaluation.md\"]"
},

{
    "location": "evaluation/#Recommendation.cross_validation",
    "page": "Evaluation",
    "title": "Recommendation.cross_validation",
    "category": "function",
    "text": "cross_validation(\n    rec_type::DataType,\n    hyperparams::Parameters,\n    da::DataAccessor,\n    n_fold::Int,\n    metric::Metric,\n    k::Int=0\n)\n\nConduct n_fold cross validation for a combination of recommender rec_type and metric metric with hyperparams. For ranking metric, accuracy is measured by top-k recommendation.\n\n\n\n\n\n"
},

{
    "location": "evaluation/#Cross-validation-1",
    "page": "Evaluation",
    "title": "Cross validation",
    "category": "section",
    "text": "cross_validation"
},

{
    "location": "evaluation/#Recommendation.RMSE",
    "page": "Evaluation",
    "title": "Recommendation.RMSE",
    "category": "type",
    "text": "RMSE\n\nRoot Mean Squared Error.\n\nmeasure(\n    metric::RMSE,\n    truth::AbstractVector,\n    pred::AbstractVector\n)\n\n\n\n\n\n"
},

{
    "location": "evaluation/#Recommendation.MAE",
    "page": "Evaluation",
    "title": "Recommendation.MAE",
    "category": "type",
    "text": "MAE\n\nMean Absolute Error.\n\nmeasure(\n    metric::MAE,\n    truth::AbstractVector,\n    pred::AbstractVector\n)\n\n\n\n\n\n"
},

{
    "location": "evaluation/#Rating-metric-1",
    "page": "Evaluation",
    "title": "Rating metric",
    "category": "section",
    "text": "RMSE\nMAE"
},

{
    "location": "evaluation/#Recommendation.Recall",
    "page": "Evaluation",
    "title": "Recommendation.Recall",
    "category": "type",
    "text": "Recall\n\nRecall@k.\n\nmeasure(\n    metric::Recall,\n    truth::Array{T},\n    pred::Array{T},\n    k::Int\n)\n\n\n\n\n\n"
},

{
    "location": "evaluation/#Recommendation.Precision",
    "page": "Evaluation",
    "title": "Recommendation.Precision",
    "category": "type",
    "text": "Precision\n\nPrecision@k.\n\nmeasure(\n    metric::Precision,\n    truth::Array{T},\n    pred::Array{T},\n    k::Int\n)\n\n\n\n\n\n"
},

{
    "location": "evaluation/#Recommendation.MAP",
    "page": "Evaluation",
    "title": "Recommendation.MAP",
    "category": "type",
    "text": "MAE\n\nMean Average Precision.\n\nmeasure(\n    metric::MAP,\n    truth::Array{T},\n    pred::Array{T}\n)\n\n\n\n\n\n"
},

{
    "location": "evaluation/#Recommendation.AUC",
    "page": "Evaluation",
    "title": "Recommendation.AUC",
    "category": "type",
    "text": "AUC\n\nArea Under the ROC Curve.\n\nmeasure(\n    metric::AUC,\n    truth::Array{T},\n    pred::Array{T}\n)\n\n\n\n\n\n"
},

{
    "location": "evaluation/#Recommendation.ReciprocalRank",
    "page": "Evaluation",
    "title": "Recommendation.ReciprocalRank",
    "category": "type",
    "text": "ReciprocalRank\n\nReciprocal Rank.\n\nmeasure(\n    metric::ReciprocalRank,\n    truth::Array{T},\n    pred::Array{T}\n)\n\n\n\n\n\n"
},

{
    "location": "evaluation/#Recommendation.MPR",
    "page": "Evaluation",
    "title": "Recommendation.MPR",
    "category": "type",
    "text": "MPR\n\nMean Percentile Rank.\n\nmeasure(\n    metric::MPR,\n    truth::Array{T},\n    pred::Array{T}\n)\n\n\n\n\n\n"
},

{
    "location": "evaluation/#Recommendation.NDCG",
    "page": "Evaluation",
    "title": "Recommendation.NDCG",
    "category": "type",
    "text": "NDCG\n\nNormalized Discounted Cumulative Gain.\n\nmeasure(\n    metric::NDCG,\n    truth::Array{T},\n    pred::Array{T},\n    k::Int\n)\n\n\n\n\n\n"
},

{
    "location": "evaluation/#Ranking-metric-1",
    "page": "Evaluation",
    "title": "Ranking metric",
    "category": "section",
    "text": "Recall\nPrecision\nMAP\nAUC\nReciprocalRank\nMPR\nNDCG"
},

]}

using Recommendation
using Test
using SparseArrays

include("test_utils.jl")

@testset "recommender" begin
    include("test_base_recommender.jl")
    include("test_compat.jl")

    @testset "baseline" begin
        include("baseline/test_user_mean.jl")
        include("baseline/test_item_mean.jl")
        include("baseline/test_most_popular.jl")
        include("baseline/test_threshold_percentage.jl")
        include("baseline/test_co_occurrence.jl")
    end

    @testset "model" begin
        include("model/test_tf_idf.jl")
        include("model/test_user_knn.jl")
        include("model/test_item_knn.jl")
        include("model/test_svd.jl")
        include("model/test_matrix_factorization.jl")
        include("model/test_factorization_machines.jl")
    end
end

@testset "evaluation" begin
    include("metrics/test_accuracy.jl")
    include("metrics/test_ranking.jl")

    include("evaluation/test_evaluate.jl")
    include("evaluation/test_cross_validation.jl")
end

@testset "data" begin
    include("test_data_accessor.jl")
    include("test_datasets.jl")
    include("test_synthetic.jl")
end

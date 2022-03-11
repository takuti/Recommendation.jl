using Recommendation
using Test
using SparseArrays

include("test_utils.jl")

# run testing modules when `test_args` is empty or `name` is in the args
# e.g. `Pkg.test("Recommendation")` => run all test cases under `@testset_default_or_if`
#      `Pkg.test("Recommendation", test_args=["misc"])` => run a test set `@testset_default_or_if "misc"` only
# reference: https://github.com/JuliaAI/MLJBase.jl/blob/4a8f3f323f91ee6b6f5fb2b3268729b3101c003c/test/runtests.jl#L52-L62
RUN_ALL_TESTS = isempty(ARGS)
macro testset_default_or_if(name, expr)
    name = string(name)
    esc(quote
        if RUN_ALL_TESTS || $name in ARGS
            @testset $name $expr
        end
    end)
end
# run testing modules if and only if `name` is explicitly specified in `test_args`
macro testset_if(name, expr)
    name = string(name)
    esc(quote
        if $name in ARGS
            @testset $name $expr
        end
    end)
end

@testset_default_or_if "recommender" begin
    include("test_base_recommender.jl")
    include("test_compat.jl")

    @testset_default_or_if "baseline" begin
        include("baseline/test_user_mean.jl")
        include("baseline/test_item_mean.jl")
        include("baseline/test_most_popular.jl")
        include("baseline/test_threshold_percentage.jl")
        include("baseline/test_co_occurrence.jl")
    end

    @testset_default_or_if "model" begin
        include("model/test_tf_idf.jl")
        include("model/test_user_knn.jl")
        include("model/test_item_knn.jl")
        include("model/test_svd.jl")
        include("model/test_matrix_factorization.jl")
        include("model/test_bpr_matrix_factorization.jl")
        include("model/test_factorization_machines.jl")
    end
end

@testset_default_or_if "evaluation" begin
    include("metrics/test_accuracy.jl")
    include("metrics/test_ranking.jl")
    include("metrics/test_intra_list.jl")
    include("metrics/test_aggregated.jl")

    include("evaluation/test_evaluate.jl")
    include("evaluation/test_cross_validation.jl")
end

@testset_default_or_if "data" begin
    include("test_data_accessor.jl")
    include("test_datasets.jl")
    include("test_synthetic.jl")
end

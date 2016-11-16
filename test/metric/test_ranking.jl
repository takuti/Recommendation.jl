function test_recall(truth::Array{Int,1}, recommend::Array{Int,1}, k::Int)
    actual = RankingMetric.recall(truth, recommend, k)
    expected = 0.333
    eps = 0.001

    @test abs(actual - expected) < eps
end

function test_precision(truth::Array{Int,1}, recommend::Array{Int,1}, k::Int)
    actual = RankingMetric.precision(truth, recommend, k)
    expected = 0.5
    eps = 0.001

    @test abs(actual - expected) < eps
end

function test_map(truth::Array{Int,1}, recommend::Array{Int,1})
    actual = RankingMetric.map(truth, recommend)
    expected = 0.756
    eps = 0.001

    @test abs(actual - expected) < eps
end

function test_auc(truth::Array{Int,1}, recommend::Array{Int,1})
    actual = RankingMetric.auc(truth, recommend)
    expected = 0.667
    eps = 0.001

    @test abs(actual - expected) < eps
end

function test_mrr(truth::Array{Int,1}, recommend::Array{Int,1})
    actual = RankingMetric.mrr(truth, recommend)
    expected = 1.0
    eps = 0.001

    @test abs(actual - expected) < eps
end

function test_mpr(truth::Array{Int,1}, recommend::Array{Int,1})
    actual = RankingMetric.mpr(truth, recommend)
    expected = 33.333
    eps = 0.001

    @test abs(actual - expected) < eps
end

function test_ndcg(truth::Array{Int,1}, recommend::Array{Int,1}, k::Int)
    actual = RankingMetric.ndcg(truth, recommend, k)
    expected = 0.613
    eps = 0.001

    @test abs(actual - expected) < eps
end

println("-- Testing ranking metrics")

truth = [1, 2, 4]
recommend = [1, 3, 2, 6, 4, 5]
k = 2

test_recall(truth, recommend, k)
test_precision(truth, recommend, k)
test_map(truth, recommend)
test_auc(truth, recommend)
test_mrr(truth, recommend)
test_mpr(truth, recommend)
test_ndcg(truth, recommend, k)

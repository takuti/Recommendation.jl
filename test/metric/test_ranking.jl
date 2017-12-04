function test_recall(truth::Array{Int,1}, pred::Array{Int,1}, k::Int)
    actual = measure(Recall(), truth, pred, k)
    expected = 0.333
    eps = 0.001

    @test abs(actual - expected) < eps
end

function test_precision(truth::Array{Int,1}, pred::Array{Int,1}, k::Int)
    actual = measure(Precision(), truth, pred, k)
    expected = 0.5
    eps = 0.001

    @test abs(actual - expected) < eps
end

function test_map(truth::Array{Int,1}, pred::Array{Int,1})
    actual = measure(MAP(), truth, pred)
    expected = 0.756
    eps = 0.001

    @test abs(actual - expected) < eps
end

function test_auc(truth::Array{Int,1}, pred::Array{Int,1})
    actual = measure(AUC(), truth, pred)
    expected = 0.667
    eps = 0.001

    @test abs(actual - expected) < eps
end

function test_mrr(truth::Array{Int,1}, pred::Array{Int,1})
    actual = measure(ReciprocalRank(), truth, pred)
    expected = 1.0
    eps = 0.001

    @test abs(actual - expected) < eps
end

function test_mpr(truth::Array{Int,1}, pred::Array{Int,1})
    actual = measure(MPR(), truth, pred)
    expected = 33.333
    eps = 0.001

    @test abs(actual - expected) < eps
end

function test_ndcg(truth::Array{Int,1}, pred::Array{Int,1}, k::Int)
    actual = measure(NDCG(), truth, pred, k)
    expected = 0.613
    eps = 0.001

    @test abs(actual - expected) < eps
end

println("-- Testing ranking metrics")

truth = [1, 2, 4]
pred = [1, 3, 2, 6, 4, 5]
k = 2

test_recall(truth, pred, k)
test_precision(truth, pred, k)
test_map(truth, pred)
test_auc(truth, pred)
test_mrr(truth, pred)
test_mpr(truth, pred)
test_ndcg(truth, pred, k)

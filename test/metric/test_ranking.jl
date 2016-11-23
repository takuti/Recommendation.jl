function test_recall(truth::Array{Int,1}, rec::Array{Int,1}, k::Int)
    actual = measure(Recall(), truth, rec, k)
    expected = 0.333
    eps = 0.001

    @test abs(actual - expected) < eps
end

function test_precision(truth::Array{Int,1}, rec::Array{Int,1}, k::Int)
    actual = measure(Precision(), truth, rec, k)
    expected = 0.5
    eps = 0.001

    @test abs(actual - expected) < eps
end

function test_map(truth::Array{Int,1}, rec::Array{Int,1})
    actual = measure(MAP(), truth, rec)
    expected = 0.756
    eps = 0.001

    @test abs(actual - expected) < eps
end

function test_auc(truth::Array{Int,1}, rec::Array{Int,1})
    actual = measure(AUC(), truth, rec)
    expected = 0.667
    eps = 0.001

    @test abs(actual - expected) < eps
end

function test_mrr(truth::Array{Int,1}, rec::Array{Int,1})
    actual = measure(MRR(), truth, rec)
    expected = 1.0
    eps = 0.001

    @test abs(actual - expected) < eps
end

function test_mpr(truth::Array{Int,1}, rec::Array{Int,1})
    actual = measure(MPR(), truth, rec)
    expected = 33.333
    eps = 0.001

    @test abs(actual - expected) < eps
end

function test_ndcg(truth::Array{Int,1}, rec::Array{Int,1}, k::Int)
    actual = measure(NDCG(), truth, rec, k)
    expected = 0.613
    eps = 0.001

    @test abs(actual - expected) < eps
end

println("-- Testing ranking metrics")

truth = [1, 2, 4]
rec = [1, 3, 2, 6, 4, 5]
k = 2

test_recall(truth, rec, k)
test_precision(truth, rec, k)
test_map(truth, rec)
test_auc(truth, rec)
test_mrr(truth, rec)
test_mpr(truth, rec)
test_ndcg(truth, rec, k)

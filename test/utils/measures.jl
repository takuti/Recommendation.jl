const truth = [1, 2, 4]
const recommend = [1, 3, 2, 6, 4, 5]
const k = 2

function test_recall()
    actual = Measures.recall(truth, recommend, k)
    expected = 0.333
    eps = 0.001

    @test abs(actual - expected) < eps
end

function test_precision()
    actual = Measures.precision(truth, recommend, k)
    expected = 0.5
    eps = 0.001

    @test abs(actual - expected) < eps
end

function test_map()
    actual = Measures.map(truth, recommend)
    expected = 0.756
    eps = 0.001

    @test abs(actual - expected) < eps
end

function test_auc()
    actual = Measures.auc(truth, recommend)
    expected = 0.667
    eps = 0.001

    @test abs(actual - expected) < eps
end

function test_mrr()
    actual = Measures.mrr(truth, recommend)
    expected = 1.0
    eps = 0.001

    @test abs(actual - expected) < eps
end

function test_mpr()
    actual = Measures.mpr(truth, recommend)
    expected = 33.333
    eps = 0.001

    @test abs(actual - expected) < eps
end

function test_ndcg()
    actual = Measures.ndcg(truth, recommend, k)
    expected = 0.613
    eps = 0.001

    @test abs(actual - expected) < eps
end

test_recall()
test_precision()
test_map()
test_auc()
test_mrr()
test_mpr()
test_ndcg()

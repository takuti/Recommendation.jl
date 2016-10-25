function test_recall()
    truth = [1, 2, 4]
    recommend = [1, 3, 2, 6, 4, 5]
    k = 2

    actual = Measures.recall(truth, recommend, k)
    expected = 0.333
    eps = 0.001

    @test abs(actual - expected) < eps
end

function test_precision()
    truth = [1, 2, 4]
    recommend = [1, 3, 2, 6, 4, 5]
    k = 2

    actual = Measures.precision(truth, recommend, k)
    expected = 0.5
    eps = 0.001

    @test abs(actual - expected) < eps
end

function test_map()
    truth = [1, 2, 4]
    recommend = [1, 3, 2, 6, 4, 5]

    actual = Measures.map(truth, recommend)
    expected = 0.756
    eps = 0.001

    @test abs(actual - expected) < eps
end

function test_auc()
    truth = [1, 2, 4]
    recommend = [1, 3, 2, 6, 4, 5]

    actual = Measures.auc(truth, recommend)
    expected = 0.667
    eps = 0.001

    @test abs(actual - expected) < eps
end

function test_mrr()
    truth = [1, 2, 4]
    recommend = [1, 3, 2, 6, 4, 5]

    actual = Measures.mrr(truth, recommend)
    expected = 1.0
    eps = 0.001

    @test abs(actual - expected) < eps
end

function test_mpr()
    truth = [1, 2, 4]
    recommend = [1, 3, 2, 6, 4, 5]

    actual = Measures.mpr(truth, recommend)
    expected = 33.333
    eps = 0.001

    @test abs(actual - expected) < eps
end

function test_ndcg()
    truth = [1, 2, 4]
    recommend = [1, 3, 2, 6, 4, 5]
    k = 2

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

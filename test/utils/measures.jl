function test_recall()
    truth = [1, 2, 4]
    recommend = [1, 3, 2, 6, 4, 5]
    k = 2

    actual = Measures.recall(truth, recommend, k)
    expected = 0.333
    eps = 0.001
    @test abs(actual - expected) < eps
end

test_recall()

function test_threshold_percentage()
    println("-- Testing ThresholdPercentage recommender")

    da = DataAccessor(sparse([1 2 3; 4 5 6]))
    recommender = ThresholdPercentage(da, Parameters(:th => 2.0))
    build(recommender)
    @test ranking(recommender, 1, 1) == 50.0
    @test ranking(recommender, 1, 2) == 100.0
end

test_threshold_percentage()

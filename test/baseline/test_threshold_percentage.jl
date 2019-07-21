function test_threshold_percentage()
    println("-- Testing ThresholdPercentage recommender")

    data = DataAccessor(sparse([1 2 3; 4 5 6]))
    recommender = ThresholdPercentage(data, 2.0)
    build!(recommender)
    @test ranking(recommender, 1, 1) == 50.0
    @test ranking(recommender, 1, 2) == 100.0
end

test_threshold_percentage()

function test_threshold_percentage(data)
    recommender = ThresholdPercentage(data, 2.0)
    fit!(recommender)
    @test predict(recommender, 1, 1) == 50.0
    @test predict(recommender, 1, 2) == 100.0

    actual = predict(recommender, [CartesianIndex(1, 1), CartesianIndex(1, 2)])
    @test actual == [50.0, 100.0]
end

println("-- Testing ThresholdPercentage recommender")
test_threshold_percentage(DataAccessor([1 2 3 nothing; 4 5 6 0]))
test_threshold_percentage(DataAccessor(sparse([1 2 3; 4 5 6])))

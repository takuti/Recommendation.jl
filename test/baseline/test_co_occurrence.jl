function test_co_occurrence(data)
    recommender = CoOccurrence(data, 1)
    fit!(recommender)
    @test predict(recommender, 1, 1) == 100.0
    @test predict(recommender, 1, 2) == 50.0
    @test predict(recommender, 1, 3) == 0.0

    actual = predict(recommender, [CartesianIndex(1, 1), CartesianIndex(1, 2), CartesianIndex(1, 3)])
    @test actual == [100.0, 50.0, 0.0]
end

println("-- Testing CoOccurrence recommender")
test_co_occurrence(DataAccessor([1 0 nothing; 4 5 0]))
test_co_occurrence(DataAccessor(sparse([1 0 0; 4 5 0])))

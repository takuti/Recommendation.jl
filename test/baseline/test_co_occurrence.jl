function test_co_occurrence(data)
    recommender = CoOccurrence(data, 1)
    build!(recommender)
    @test ranking(recommender, 1, 1) == 100.0
    @test ranking(recommender, 1, 2) == 50.0
    @test ranking(recommender, 1, 3) == 0.0
end

println("-- Testing CoOccurrence recommender")
test_co_occurrence(DataAccessor([1 0 missing; 4 5 0]))
test_co_occurrence(DataAccessor(sparse([1 0 0; 4 5 0])))

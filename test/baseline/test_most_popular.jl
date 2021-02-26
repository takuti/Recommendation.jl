function test_most_popular()
    println("-- Testing MostPopular recommender")

    data = DataAccessor(sparse([1 2 3; 4 5 0]))
    recommender = MostPopular(data)
    build!(recommender)
    @test ranking(recommender, 1, 1) == 2.0
    @test ranking(recommender, 1, 3) == 1.0

    n_user, n_item = 5, 10
    events = [Event(1, 2, 1), Event(3, 2, 1), Event(2, 6, 4)]
    data = DataAccessor(events, n_user, n_item)
    recommender = MostPopular(data)
    build!(recommender)
    @test ranking(recommender, 1, 1) == 0.0
    @test ranking(recommender, 1, 2) == 2.0
    @test ranking(recommender, 1, 6) == 1.0
end

test_most_popular()

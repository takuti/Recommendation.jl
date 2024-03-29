function test_most_popular()
    println("-- Testing MostPopular recommender")

    data = DataAccessor([1 2 3; 4 5 nothing])
    recommender = MostPopular(data)
    fit!(recommender)
    @test predict(recommender, 1, 1) == 2.0
    @test predict(recommender, 1, 3) == 1.0

    data = DataAccessor(sparse([1 2 3; 4 5 0]))
    recommender = MostPopular(data)
    fit!(recommender)
    @test predict(recommender, 1, 1) == 2.0
    @test predict(recommender, 1, 3) == 1.0

    n_users, n_items = 5, 10
    events = [Event(1, 2, 1), Event(3, 2, 1), Event(2, 6, 4)]
    data = DataAccessor(events, n_users, n_items)
    recommender = MostPopular(data)
    fit!(recommender)
    @test predict(recommender, 1, 1) == 0.0
    @test predict(recommender, 1, 2) == 2.0
    @test predict(recommender, 1, 6) == 1.0
end

test_most_popular()

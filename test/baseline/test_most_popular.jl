function test_most_popular()
    println("-- Testing MostPopular recommender")

    recommender = MostPopular([1 2 3; 4 5 0])
    @test ranking(recommender, 1, 1) == 2.0
    @test ranking(recommender, 1, 3) == 1.0
end

test_most_popular()

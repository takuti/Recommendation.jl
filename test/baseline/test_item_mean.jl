function test_user_mean()
    println("-- Testing UserMean recommender")

    recommender = UserMean(sparse([1 2 3; 4 5 6]))
    actual = predict(recommender, 1, 1)

    @test actual == 2.0
end

test_user_mean()

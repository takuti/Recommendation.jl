function test_user_mean(data)
    recommender = UserMean(data)
    fit!(recommender)
    actual = predict(recommender, 1, 1)

    @test actual == 2.0
end

println("-- Testing UserMean recommender")
test_user_mean(DataAccessor([1 2 3 nothing 4; 4 5 6 7 8]))
test_user_mean(DataAccessor(sparse([1 2 3; 4 5 6])))

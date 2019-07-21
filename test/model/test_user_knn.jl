function test_user_knn()
    println("-- Testing UserKNN rating prediction")

    m = [NaN 3 NaN 1 2 1 NaN 4
         1 2 NaN NaN 3 2 NaN 3
         NaN 2 3 3 NaN 5 NaN 1]
    data = DataAccessor(m)

    recommender = UserKNN(data, 1)
    build!(recommender)

    @test isapprox(recommender.sim[1, 2], 0.447, atol=1e-3)
    @test isapprox(recommender.sim[2, 3], -0.693, atol=1e-3)

    rec = recommend(recommender, 1, 3, [i for i in 1:8])
    @test rec[1] == (5 => 3.0)
    @test rec[2] == (8 => 3.0)
    @test rec[3] == (2 => 2.0)

    recommender = UserKNN(data, 1, true)
    build!(recommender)

    @test isapprox(recommender.sim[1, 2], 0.447, atol=1e-3)
    @test isapprox(recommender.sim[2, 3], -0.693, atol=1e-3)

    rec = recommend(recommender, 1, 3, [i for i in 1:8])
    @test rec[1] == (5 => 3.0)
    @test rec[2] == (8 => 3.0)
    @test rec[3] == (7 => 2.2)
end

test_user_knn()

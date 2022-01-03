function test_user_knn()
    println("-- Testing UserKNN rating prediction")

    m = [missing 3 missing 1 2 1 missing 4
         1 2 missing missing 3 2 missing 3
         missing 2 3 3 missing 5 missing 1]
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
    @test rec[3] == (3 => 2.2)
end

test_user_knn()

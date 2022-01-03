function test_user_knn(v)
    m = [v 3 v 1 2 1 v 4
         1 2 v v 3 2 v 3
         v 2 3 3 v 5 v 1]
    data = DataAccessor(ismissing(v) ? m : sparse(m))

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

println("-- Testing UserKNN rating prediction")
test_user_knn(missing)
test_user_knn(0)

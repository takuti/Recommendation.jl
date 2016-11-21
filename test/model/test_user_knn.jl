function test_user_knn()
    println("-- Testing UserKNN rating prediction")

    m = [NaN 3 NaN 1 2 1 NaN 4
         1 2 NaN NaN 3 2 NaN 3
         NaN 2 3 3 NaN 5 NaN 1]
    da = DataAccessor(m)

    recommender = UserKNN(da, 1)
    build(recommender)

    @test_approx_eq_eps recommender.sim[1, 2] 0.447 1e-3
    @test_approx_eq_eps recommender.sim[2, 3] -0.693 1e-3

    rec = execute(recommender, 1, 3, [i for i in 1:8])
    @test rec[1] == (5 => 3.0)
    @test rec[2] == (8 => 3.0)
    @test rec[3] == (2 => 2.0)

    recommender = UserKNN(da, 1, is_normalized=true);
    build(recommender)

    @test_approx_eq_eps recommender.sim[1, 2] 0.447 1e-3
    @test_approx_eq_eps recommender.sim[2, 3] -0.693 1e-3

    rec = execute(recommender, 1, 3, [i for i in 1:8])
    @test rec[1] == (5 => 3.0)
    @test rec[2] == (8 => 3.0)
    @test rec[3] == (7 => 2.2)
end

test_user_knn()

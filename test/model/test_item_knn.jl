function test_item_knn()
    println("-- Testing ItemKNN rating prediction")

    m = [NaN 3 NaN 1 2 1 NaN 4
         1 2 NaN NaN 3 2 NaN 3
         NaN 2 3 3 NaN 5 NaN 1]
    da = DataAccessor(m)

    recommender = ItemKNN(da, 1)
    build(recommender)
    @test_approx_eq_eps recommender.sim[1, 2] 0.485 1e-3
    @test_approx_eq_eps recommender.sim[5, 6] 0.405 1e-3

    rec = execute(recommender, 1, 4, [i for i in 1:8])
    @test rec[1] == (8 => 4.0)
    @test rec[2] == (2 => 3.0)
    @test rec[3] == (5 => 2.0)
    @test rec[4] == (4 => 1.0)

    recommender = ItemKNN(da, 1)
    build(recommender, is_adjusted_cosine=true)
    @test_approx_eq_eps recommender.sim[1, 2] 0.174 1e-3
    @test_approx_eq_eps recommender.sim[5, 6] 0.039 1e-3

    rec = execute(recommender, 1, 4, [i for i in 1:8])
    @test rec[1] == (8 => 4.0)
    @test rec[2] == (2 => 3.0)
    @test rec[3] == (5 => 2.0)
    @test rec[4] == (4 => 1.0)
end

test_item_knn()

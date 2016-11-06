function test_user_knn()
    println("-- Testing UserKNN rating prediction")

    m = [NaN 3 NaN 1 2 1 NaN 4
         1 2 NaN NaN 3 2 NaN 3
         NaN 2 3 3 NaN 5 NaN 1]

    recommender = UserKNN(sparse(m), 1, similarity="pearson");

    @test_approx_eq_eps recommender.sim[1, 2] 0.447 1e-3
    @test_approx_eq_eps recommender.sim[2, 3] -0.693 1e-3

    rec = execute(recommender, 1, 3, ["item$(i)" for i in 1:8])
    @test rec[1] == ("item8" => 3.0)
    @test rec[2] == ("item5" => 3.0)
    @test rec[3] == ("item6" => 2.0)

    recommender = UserKNN(sparse(m), 1, similarity="pearson", is_normalized_pred=true);

    @test_approx_eq_eps recommender.sim[1, 2] 0.447 1e-3
    @test_approx_eq_eps recommender.sim[2, 3] -0.693 1e-3

    rec = execute(recommender, 1, 3, ["item$(i)" for i in 1:8])
    @test rec[1] == ("item8" => 3.0)
    @test rec[2] == ("item5" => 3.0)
    @test rec[3] == ("item4" => 2.2)
end

test_user_knn()

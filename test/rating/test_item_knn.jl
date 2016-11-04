function test_item_knn()
    println("-- Testing ItemKNN rating prediction")

    m = [NaN 3 NaN 1 2 1 NaN 4
         1 2 NaN NaN 3 2 NaN 3
         NaN 2 3 3 NaN 5 NaN 1]

    recommender = ItemKNN(sparse(m), 1);
    @test_approx_eq_eps recommender.sim[1, 2] 0.485 1e-3
    @test_approx_eq_eps recommender.sim[5, 6] 0.405 1e-3

    rec = execute(recommender, 1, 4, ["item$(i)" for i in 1:8])
    @test first(rec[1]) == "item8"
    @test last(rec[1]) == 4.0
    @test first(rec[2]) == "item2"
    @test last(rec[2]) == 3.0
    @test first(rec[3]) == "item5"
    @test last(rec[3]) == 2.0
    @test first(rec[4]) == "item6"
    @test last(rec[4]) == 1.0

    recommender = ItemKNN(sparse(m), 1, is_normalized=true);
    @test_approx_eq_eps recommender.sim[1, 2] 0.174 1e-3
    @test_approx_eq_eps recommender.sim[5, 6] 0.039 1e-3

    rec = execute(recommender, 1, 4, ["item$(i)" for i in 1:8])
    @test first(rec[1]) == "item8"
    @test last(rec[1]) == 4.0
    @test first(rec[2]) == "item2"
    @test last(rec[2]) == 3.0
    @test first(rec[3]) == "item5"
    @test last(rec[3]) == 2.0
    @test first(rec[4]) == "item6"
    @test last(rec[4]) == 1.0
end

test_item_knn()

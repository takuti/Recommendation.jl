function test_item_knn()
    println("-- Testing ItemKNN rating prediction")

    m = [missing 3 missing 1 2 1 missing 4
         1 2 missing missing 3 2 missing 3
         missing 2 3 3 missing 5 missing 1]
    data = DataAccessor(m)

    recommender = ItemKNN(data, 1)
    build!(recommender)
    @test isapprox(recommender.sim[1, 2], 0.485, atol=1e-3)
    @test isapprox(recommender.sim[5, 6], 0.405, atol=1e-3)

    rec = recommend(recommender, 1, 4, [i for i in 1:8])
    @test rec[1] == (8 => 4.0)
    @test rec[2] == (2 => 3.0)
    @test rec[3] == (5 => 2.0)
    @test rec[4] == (4 => 1.0)

    recommender = ItemKNN(data, 1)
    build!(recommender, adjusted_cosine=true)
    @test isapprox(recommender.sim[1, 2], 0.174, atol=1e-3)
    @test isapprox(recommender.sim[5, 6], 0.039, atol=1e-3)

    rec = recommend(recommender, 1, 4, [i for i in 1:8])
    @test rec[1] == (8 => 4.0)
    @test rec[2] == (2 => 3.0)
    @test rec[3] == (5 => 2.0)
    @test rec[4] == (4 => 1.0)
end

test_item_knn()

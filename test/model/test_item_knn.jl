function test_item_knn(v)
    m = [v 3 v 1 2 1 v 4
         1 2 v v 3 2 v 3
         v 2 3 3 v 5 v 1]
    data = DataAccessor(isa(v, Unknown) ? m : sparse(m))

    recommender = ItemKNN(data, 1)
    fit!(recommender)
    @test isapprox(recommender.sim[1, 2], 0.485, atol=1e-3)
    @test isapprox(recommender.sim[5, 6], 0.405, atol=1e-3)

    rec = recommend(recommender, 1, 4, [i for i in 1:8])
    @test rec[1] == (8 => 4.0)
    @test rec[2] == (2 => 3.0)
    @test rec[3] == (5 => 2.0)
    @test rec[4] == (4 => 1.0)

    recommender = ItemKNN(data, 1)
    fit!(recommender, adjusted_cosine=true)
    @test isapprox(recommender.sim[1, 2], 0.174, atol=1e-3)
    @test isapprox(recommender.sim[5, 6], 0.039, atol=1e-3)

    rec = recommend(recommender, 1, 4, [i for i in 1:8])
    @test rec[1] == (8 => 4.0)
    @test rec[2] == (2 => 3.0)
    @test rec[3] == (5 => 2.0)
    @test rec[4] == (4 => 1.0)
end

println("-- Testing ItemKNN rating prediction")
test_item_knn(nothing)
test_item_knn(0)

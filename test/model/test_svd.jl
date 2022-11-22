function test_svd()
    println("-- Testing SVD-based recommender")

    m = [nothing 3 nothing 1 2 1 nothing 4
         1 2 nothing nothing 3 2 nothing 3
         nothing 2 3 3 nothing 5 nothing 1]
    data = DataAccessor(m)

    recommender = SVD(data, 2)
    fit!(recommender)

    # dimensionality reduction should preserve user-user/item-item similarities
    # i.e., recommendation list should be same as ItemKNN
    rec = recommend(recommender, 1, 4, [i for i in 1:8])
    @test first(rec[1]) == 8
    @test first(rec[2]) == 2
    @test first(rec[3]) == 5
    @test first(rec[4]) == 6
    @test predict(recommender, 1, 2) > predict(recommender, 1, 5)
end

test_svd()

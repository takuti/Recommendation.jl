function test_svd()
    println("-- Testing SVD-based recommender")

    m = [missing 3 missing 1 2 1 missing 4
         1 2 missing missing 3 2 missing 3
         missing 2 3 3 missing 5 missing 1]
    data = DataAccessor(m)

    recommender = SVD(data, 2)
    build!(recommender)

    # dimensionality reduction should preserve user-user/item-item similarities
    # i.e., recommendation list should be same as ItemKNN
    rec = recommend(recommender, 1, 4, [i for i in 1:8])
    @test first(rec[1]) == 8
    @test first(rec[2]) == 2
    @test first(rec[3]) == 5
    @test first(rec[4]) == 6
end

test_svd()

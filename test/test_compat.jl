function test_build()
    println("-- Testing a deprecated `build!` function to check compatibility")

    m = [0 3 0 1 2 1 0 4
         1 2 0 0 3 2 0 3
         0 2 3 3 0 5 0 1]
    data = DataAccessor(sparse(m))

    recommender = MatrixFactorization(data, 2)

    # make sure build! works as a synonym of fit!
    build!(recommender, learning_rate=15e-4, max_iter=100)

    # top-4 recommended item set should be same as CF/SVD-based recommender
    rec = recommend(recommender, 1, 4, [i for i in 1:8])
    @test Set([item for (item, score) in rec]) == Set([2, 5, 6, 8])
end

test_build()

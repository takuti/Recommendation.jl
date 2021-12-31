function run(recommender::Type{T}) where {T<:Recommender}
    m = [NaN 3 NaN 1 2 1 NaN 4
         1 2 NaN NaN 3 2 NaN 3
         NaN 2 3 3 NaN 5 NaN 1]
    data = DataAccessor(m)

    recommender = recommender(data, 2)
    build!(recommender, learning_rate=15e-4, max_iter=100)

    # top-4 recommended item set should be same as CF/SVD-based recommender
    rec = recommend(recommender, 1, 4, [i for i in 1:8])
    @test Set([item for (item, score) in rec]) == Set([2, 5, 6, 8])
end

function test_mf()
    println("-- Testing MF-based (aliased) recommender")
    run(MF)
end

function test_matrix_factorization()
    println("-- Testing Matrix Factorization-based recommender")
    run(MatrixFactorization)
end

function test_mf_with_random_init()
    println("-- Testing MF-based recommender with randomly initialized params")
    m = [NaN 3 NaN 1 2 1 NaN 4
         1 2 NaN NaN 3 2 NaN 3
         NaN 2 3 3 NaN 5 NaN 1]
    data = DataAccessor(m)

    recommender = MF(data, 2)
    build!(recommender, random_init=true)

    rec = recommend(recommender, 1, 4, [i for i in 1:8])
    @test size(rec, 1) == 4  # top-4 recos
end

test_mf()
test_matrix_factorization()
test_mf_with_random_init()

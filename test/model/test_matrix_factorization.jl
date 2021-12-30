function run(recommender::Type{T}) where {T<:Recommender}
    m = [NaN 3 NaN 1 2 1 NaN 4
         1 2 NaN NaN 3 2 NaN 3
         NaN 2 3 3 NaN 5 NaN 1]
    data = DataAccessor(m)

    recommender = recommender(data, 2)
    build!(recommender, learning_rate=15e-4, max_iter=100)

    # top-4 recommantion list should be same as CF/SVD-based recommender
    rec = recommend(recommender, 1, 4, [i for i in 1:8])
    @test Set([first(r) for r in rec]) == Set([2, 5, 6, 8])
end

function test_mf()
    println("-- Testing MF-based (aliased) recommender")
    run(MF)
end

function test_matrix_factorization()
    println("-- Testing Matrix Factorization-based recommender")
    run(MatrixFactorization)
end

test_mf()
test_matrix_factorization()

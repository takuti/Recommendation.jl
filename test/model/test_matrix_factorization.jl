function run(recommender::Type{T}, v) where {T<:Recommender}
    m = [v 3 v 1 2 1 v 4
         1 2 v v 3 2 v 3
         v 2 3 3 v 5 v 1]
    data = DataAccessor(ismissing(v) ? m : sparse(m))

    recommender = recommender(data, 2)
    build!(recommender, learning_rate=15e-4, max_iter=100)

    # top-4 recommended item set should be same as CF/SVD-based recommender
    rec = recommend(recommender, 1, 4, [i for i in 1:8])
    @test Set([item for (item, score) in rec]) == Set([2, 5, 6, 8])
end

function test_mf()
    println("-- Testing MF-based (aliased) recommender")
    run(MF, missing)
    run(MF, 0)
end

function test_matrix_factorization()
    println("-- Testing Matrix Factorization-based recommender")
    run(MatrixFactorization, missing)
    run(MatrixFactorization, 0)
end

function test_mf_with_random_init(v)
    m = [v 3 v 1 2 1 v 4
         1 2 v v 3 2 v 3
         v 2 3 3 v 5 v 1]
    data = DataAccessor(ismissing(v) ? m : sparse(m))

    recommender = MF(data, 2)
    build!(recommender, random_init=true)

    rec = recommend(recommender, 1, 4, [i for i in 1:8])
    @test size(rec, 1) == 4  # top-4 recos
end

test_mf()
test_matrix_factorization()

println("-- Testing MF-based recommender with randomly initialized params")
test_mf_with_random_init(missing)
test_mf_with_random_init(0)

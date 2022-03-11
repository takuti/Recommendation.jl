function run(recommender::Type{T}, v) where {T<:Recommender}
    m = [v 3 v 1 2 1 v 4
         1 2 v v 3 2 v 3
         v 2 3 3 v 5 v 1]
    data = DataAccessor(isa(v, Unknown) ? m : sparse(m))

    recommender = recommender(data, 2)
    fit!(recommender, learning_rate=15e-4, max_iter=100, bootstrap_sampling=false)

    # top-4 recommended item set should be same as CF/SVD-based recommender
    rec = recommend(recommender, 1, 4, [i for i in 1:8])
    @test Set([item for (item, score) in rec]) == Set([2, 5, 6, 8])
end

function test_bprmf()
    println("-- Testing BPRMF-based (aliased) recommender")
    run(BPRMF, nothing)
    run(BPRMF, 0)
end

function test_bpr_matrix_factorization()
    println("-- Testing BPR Matrix Factorization-based recommender")
    run(BPRMatrixFactorization, nothing)
    run(BPRMatrixFactorization, 0)
end

function test_bprmf_with_random_init(v)
    m = [v 3 v 1 2 1 v 4
         1 2 v v 3 2 v 3
         v 2 3 3 v 5 v 1]
    data = DataAccessor(isa(v, Unknown) ? m : sparse(m))

    recommender = BPRMF(data, 2)
    fit!(recommender, random_init=true)

    rec = recommend(recommender, 1, 4, [i for i in 1:8])
    @test size(rec, 1) == 4  # top-4 recos
end

test_bprmf()
test_bpr_matrix_factorization()

println("-- Testing BPR MF-based recommender with randomly initialized params")
test_bprmf_with_random_init(nothing)
test_bprmf_with_random_init(0)

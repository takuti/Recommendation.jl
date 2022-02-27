function test_tf_idf(sparsify::Bool)
    m = [1  0  1  0  1  1  0  0  0  1
         0  1  1  1  0  0  0  1  0  0
         0  0  0  1  1  1  0  0  0  0
         0  0  1  1  0  0  1  1  0  0
         0  1  0  0  0  0  0  0  1  1
         1  0  0  1  0  0  0  0  0  0
         0  0  0  0  0  0  0  1  0  1
         0  0  1  1  0  0  1  0  0  1
         0  0  0  0  0  1  0  0  1  0
         0  1  0  0  1  0  1  0  0  0
         0  0  1  0  1  0  0  0  1  0
         1  0  0  0  0  1  1  0  0  0
         0  0  1  1  1  0  0  1  0  0
         0  1  1  1  0  0  0  0  1  0
         0  0  0  1  0  1  1  1  0  0
         1  0  0  0  0  1  0  0  1  0
         0  1  1  1  0  0  0  1  0  0
         0  0  0  1  0  0  0  0  1  0
         0  1  1  0  1  0  1  0  0  1
         0  0  1  1  0  0  1  0  1  0]

    tf = sum(m, dims=1)
    idf = 1 ./ tf

    users = [ 1  -1
             -1   1
              0   0
              0   1
              0   0
              1   0
              0   0
              0   0
              0   0
              0   0
              0   0
              0  -1
              0   0
              0   0
              0   0
              1   0
              0   1
              0   0
             -1   0
              0   0]
    if sparsify
        m = sparse(m)
    end
    uv1 = sparse(users[:, 1])
    uv2 = sparse(users[:, 2])

    n_docs = size(m, 1)
    docs = [i for i in 1:20]


    # Case: basic
    data = DataAccessor(m)
    set_user_attribute(data, 1, uv1)
    set_user_attribute(data, 2, uv2)

    recommender = TFIDF(data, tf, ones(size(tf))) # do not use IDF
    rec = recommend(recommender, 1, n_docs, docs)
    @test first(rec[2]) == 1
    @test first(rec[3]) == 12
    @test last(rec[2]) == 4 && last(rec[3]) == 4


    # Case: normalized matrix
    m_normalized = m ./ sqrt.(sum(m.^2, dims=2))
    data = DataAccessor(m_normalized)
    set_user_attribute(data, 1, uv1)
    set_user_attribute(data, 2, uv2)

    recommender = TFIDF(data, tf, ones(size(tf))) # do not use IDF
    rec = recommend(recommender, 1, n_docs, docs)
    @test first(rec[5]) == 1
    @test isapprox(last(rec[5]), 1.0090, atol=1e-4)
    rec = recommend(recommender, 2, n_docs, docs)
    @test first(rec[10]) == 7
    @test isapprox(last(rec[10]), 0.7444, atol=1e-4)
    @test first(rec[11]) == 19
    @test isapprox(last(rec[11]), 0.4834, atol=1e-4)


    # Case: normalized matrix & using IDF
    recommender = TFIDF(data, tf, idf)
    rec = recommend(recommender, 1, n_docs, docs)
    @test first(rec[4]) == 1
    @test isapprox(last(rec[4]), 0.2476, atol=1e-4)
end

println("-- Testing TF-IDF content-based recommender")
test_tf_idf(true)
test_tf_idf(false)

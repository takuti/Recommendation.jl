function test_tf_idf()
    println("-- Testing TF-IDF content-based recommender")

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
    sm = sparse(m)

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
    uv1 = sparse(users[:, 1])
    uv2 = sparse(users[:, 2])

    n_doc = size(m, 1)
    doc_names = [string("doc", i) for i in 1:20];

    # Case: basic
    recommender = TFIDF(sm)
    rec = execute(recommender, uv1, n_doc, doc_names)
    @test first(rec[2]) == "doc1"
    @test first(rec[3]) == "doc12"
    @test last(rec[2]) == 4 && last(rec[3]) == 4

    # Case: normalized matrix
    recommender = TFIDF(sm, is_normalized=true)
    rec = execute(recommender, uv1, n_doc, doc_names)
    @test first(rec[5]) == "doc1"
    @test_approx_eq_eps last(rec[5]) 1.0090 1e-4
    rec = execute(recommender, uv2, n_doc, doc_names)
    @test first(rec[10]) == "doc7"
    @test_approx_eq_eps last(rec[10]) 0.7444 1e-4
    @test first(rec[11]) == "doc19"
    @test_approx_eq_eps last(rec[11]) 0.4834 1e-4

    # Case: normalized matrix & using IDF
    recommender = TFIDF(sm, does_use_idf=true, is_normalized=true)
    rec = execute(recommender, uv1, n_doc, doc_names)
    @test first(rec[4]) == "doc1"
    @test_approx_eq_eps last(rec[4]) 0.2476 1e-4
end

test_tf_idf()

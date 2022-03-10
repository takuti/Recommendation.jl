function test_onehot_value()
    println("-- Testing onehot encoding of a single value")

    @test_throws ErrorException onehot(1, [1, 1, 2, 3])
    @test_throws ErrorException onehot(0, [1, 2, 3])

    value_set = ["Male", "Female", "Others", nothing, missing]
    @test onehot("Male", value_set) == [1, 0, 0]
    @test onehot("Female", value_set) == [0, 1, 0]
    @test onehot("Others", value_set) == [0, 0, 1]
    @test onehot(nothing, value_set) == [0, 0, 0]
    @test onehot(missing, value_set) == [0, 0, 0]

end

function test_onehot_vector()
    println("-- Testing onehot encoding of a categorical vector")

    m = ["Male", nothing, "Female", missing, "Others"]
    expected = [1 0 0
                0 0 0
                0 1 0
                0 0 0
                0 0 1]
    @test onehot(m) == expected
end

function test_onehot_matrix()
    println("-- Testing onehot encoding of a matrix")

    m = ["Male"   2
         "Female" 3
         "Others" missing
         nothing  1]
    m_encoded = onehot(m)
    expected = [1 0 0 1 0 0
                0 1 0 0 1 0
                0 0 1 0 0 0
                0 0 0 0 0 1]
    @test m_encoded == expected
end

function test_binarize_multi_label()
    println("-- Testing multi-label binarization")
    v = ["Comedy", nothing, "Action", missing, "Anime"]
    @test binarize_multi_label(v, ["Action", "Anime", "Bollywood", "Comedy"]) == [1, 1, 0, 1]
    @test_throws ErrorException binarize_multi_label([1, 2, 3, 4], [1, 1, 2, 3, 4])
end

function test_get_ranked_triples()
    println("-- Testing ranked triples generator")
    R = [1 0 3 0
         0 2 3 4]
    @test sorted(get_ranked_triples(R)) == sorted([
        (1, 1, 2), (1, 1, 4), (1, 3, 2), (1, 3, 4),
        (2, 2, 1), (2, 3, 1), (2, 4, 1)
    ])
end

test_onehot_value()
test_onehot_vector()
test_onehot_matrix()
test_binarize_multi_label()
test_get_ranked_triples()

function test_onehot_value()
    println("-- Testing onehot encoding of a single value")

    @test_throws ErrorException onehot(1, [1, 1, 2, 3])
    @test_throws ErrorException onehot(0, [1, 2, 3])

    value_set = ["Male", "Female", "Others", nothing, missing]
    @test onehot("Male", value_set) == [1, 0, 0]
    @test onehot("Female", value_set) == [0, 1, 0]
    @test onehot("Others", value_set) == [0, 0, 1]
    @test onehot(nothing, value_set) == [0, 0, 0]
    @test onehot(nothing, value_set) == [0, 0, 0]

end

function test_onehot_vector()
    println("-- Testing onehot encoding of a categorical vector")

    m = ["Male", "Female", "Others", nothing, missing]
    expected = [1 0 0
                0 1 0
                0 0 1
                0 0 0
                0 0 0]
    @test onehot(m) == expected
end

test_onehot_value()
test_onehot_vector()

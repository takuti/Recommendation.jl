function test_onehot_value()
    println("-- Testing onehot encoding of a single value")
    value_set = ["Male", "Female", nothing]
    @test  onehot("Male", value_set) == [1, 0]
    @test  onehot("Female", value_set) == [0, 1]
    @test  onehot(nothing, value_set) == [0, 0]
end

function test_onehot_vector()
    m = ["Male", "Female", nothing]
    expected = [1 0
                0 1
                0 0]
    @test onehot(m) == expected
end

test_onehot_value()
test_onehot_vector()

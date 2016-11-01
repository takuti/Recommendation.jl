immutable Foo <: Recommender
    m::SparseMatrixCSC
end

function test_not_implemented_error()
    println("-- Testing Recommender base type")
    recommender = Foo(sparse([1 2 3; 4 5 6]))
    @test_throws ErrorException predict(recommender, 1, 1)
    @test_throws ErrorException ranking(recommender, 1, 1)
end

test_not_implemented_error()

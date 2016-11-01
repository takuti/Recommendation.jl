immutable Foo <: Recommender
    mat::Array{Float64,2}
end

function test_not_implemented_error()
    recommender = Foo([1 2 3; 4 5 6])
    @test_throws ErrorException predict(recommender, 1, 1)
end

test_not_implemented_error()

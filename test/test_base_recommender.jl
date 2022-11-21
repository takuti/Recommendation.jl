struct Foo <: Recommender
    data::DataAccessor
end

function test_not_implemented_error()
    println("-- Testing recommender not implemented case")
    data = DataAccessor(sparse([1 2 3; 4 5 6]))
    recommender = Foo(data)
    @test_throws ErrorException fit!(recommender)
    @test_throws ErrorException predict(recommender, 1, 1)
    @test_throws ErrorException predict(recommender, 1, 1)
end

function test_not_build_error()
    println("-- Testing recommender not built case")

    # non-personalized (MostPopular) recommendation for 3 items
    data = DataAccessor(sparse([1 0 0; 4 5 0]))
    recommender = MostPopular(data)

    # fit!(recommender) <- should be called before recommend()

    @test_throws ErrorException recommend(recommender, 1, 3, [1, 2, 3])
end

function test_recommend()
    println("-- Testing recommender execution")

    # non-personalized (MostPopular) recommendation for 3 items
    data = DataAccessor(sparse([1 0 0; 4 5 0]))
    recommender = MostPopular(data)
    fit!(recommender)
    pairs = recommend(recommender, 1, 3, [1, 2, 3])

    @test first(pairs[1]) == 1
    @test last(pairs[1]) == 2
    @test first(pairs[2]) == 2
    @test last(pairs[2]) == 1
    @test first(pairs[3]) == 3
    @test last(pairs[3]) == 0

    @test predict(recommender, 1, 3) == 0
    @test predict(recommender, [CartesianIndex(1, 3), CartesianIndex(2, 2), CartesianIndex(3, 1)]) == [0, 1, 2]
end

function test_data_size_validation()
    println("-- Testing validation that compares recommnder's bundled data with another")

    data = DataAccessor(sparse([1 0 0; 4 5 0]))
    recommender = MostPopular(data)
    fit!(recommender)

    validate(recommender, data)  # to pass
    @test_throws ErrorException validate(recommender, DataAccessor([1 2 3]))
    @test_throws ErrorException validate(recommender, DataAccessor([1 2; 3 4]))
end

test_not_build_error()
test_not_implemented_error()
test_recommend()
test_data_size_validation()

function test_item_mean(data)
    recommender = ItemMean(data)
    fit!(recommender)
    actual = predict(recommender, 1, 1)

    @test actual == 2.5
end

println("-- Testing ItemMean recommender")
test_item_mean(DataAccessor([5 2 3; nothing 4 6]))
test_item_mean(DataAccessor(sparse([1 2 3; 4 5 6])))

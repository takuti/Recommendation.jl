function test_item_mean()
    println("-- Testing ItemMean recommender")

    recommender = ItemMean([1 2 3; 4 5 6])
    actual = predict(recommender, 1, 1)

    @test actual == 2.5
end

test_item_mean()

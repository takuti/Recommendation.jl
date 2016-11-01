const mat = [1 2 3; 4 5 6]

function test_user_mean()
    recommender = UserMean(mat)
    actual = predict(recommender, 1, 1)

    @test actual == 2.0
end

function test_item_mean()
    recommender = ItemMean(mat)
    actual = predict(recommender, 1, 1)

    @test actual == 2.5
end

test_user_mean()
test_item_mean()

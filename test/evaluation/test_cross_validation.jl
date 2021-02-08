function test_cross_validation_accuracy()
    println("-- Testing cross validation with accuracy metrics")

    m = [NaN 3 NaN 1 2 1 NaN 4
         1 2 NaN NaN 3 2 NaN 3
         NaN NaN NaN NaN NaN NaN NaN NaN
         NaN 2 3 3 NaN 5 NaN 1]
    data = DataAccessor(m)

    fold = 5

    # MF(data, 2)
    @test cross_validation(fold, MAE, MF, data, 2) <= 0.5

    # UserMean(data)
    @test cross_validation(fold, MAE, UserMean, data) <= 2.5

    # ItemMean(data)
    @test cross_validation(fold, MAE, ItemMean, data) <= 2.5

    # UserKNN(data, 2, true)
    @test cross_validation(fold, MAE, UserKNN, data, 2, true) <= 0.5
end

function test_cross_validation_ranking()
    println("-- Testing cross validation with ranking metrics")

    m = [NaN 3 NaN 1 2 1 NaN 4
         1 2 NaN NaN 3 2 NaN 3
         NaN 2 3 3 NaN 5 NaN 1]
    data = DataAccessor(m)

    # 5-fold, top-4 recommendation
    fold = 5
    k = 4

    # MF(data, 2)
    @test cross_validation(fold, Recall, k, MF, data, 2) <= 0.5

    # MostPopular(data)
    @test cross_validation(fold, Recall, k, MostPopular, data) <= 0.5

    # UserKNN(data, 2, true)
    @test cross_validation(fold, Recall, k, UserKNN, data, 2, true) <= 0.5
end

test_cross_validation_accuracy()
test_cross_validation_ranking()

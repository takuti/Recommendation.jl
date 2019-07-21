function test_cross_validation()
    println("-- Testing cross validation")

    m = [NaN 3 NaN 1 2 1 NaN 4
         1 2 NaN NaN 3 2 NaN 3
         NaN 2 3 3 NaN 5 NaN 1]
    da = DataAccessor(m)

    # 5-fold, top-4 recommendation with MF(da, 2)
    @test cross_validation(5, Recall, 4, MF, da, 2) <= 0.5

    # MostPopular(da)
    @test cross_validation(5, Recall, 4, MostPopular, da) <= 0.5

    # UserKNN(da, 2, true)
    @test cross_validation(5, Recall, 4, UserKNN, da, 2, true) <= 0.5
end

test_cross_validation()

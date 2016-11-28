function test_cross_validation()
    println("-- Testing cross validation")

    m = [NaN 3 NaN 1 2 1 NaN 4
         1 2 NaN NaN 3 2 NaN 3
         NaN 2 3 3 NaN 5 NaN 1]
    da = DataAccessor(m)

    @test cross_validation(MF, Parameters(:k => 2), da, 5, # 5-fold
                           Recall(), 4) <= 0.5
end

test_cross_validation()

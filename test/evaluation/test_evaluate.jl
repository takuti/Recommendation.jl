function test_evaluate()
    println("-- Testing evaluate function")

    m = [NaN 3 NaN 1 2 1 NaN 4
         1 2 NaN NaN 3 2 NaN 3
         NaN 2 3 3 NaN 5 NaN 1]
    da = DataAccessor(m)
    recommender = MF(da, 2)

    truth_m = [1 3 4 1 2 1 2 4
               1 2 4 1 3 2 2 3
               5 2 3 3 4 5 2 1]
    truth_da = DataAccessor(truth_m)

    # average error should be less than 2.5 in the 1-to-5 rating
    @test evaluate(recommender, truth_da, RMSE()) < 2.5
end

test_evaluate()

function test_evaluate_explicit()
    println("-- Testing evaluate function for explicit feedback")

    m = [NaN 3 NaN 1 2 1 NaN 4
         1 2 NaN NaN 3 2 NaN 3
         NaN 2 3 3 NaN 5 NaN 1]
    data = DataAccessor(m)
    recommender = MF(data, 2)
    build!(recommender)

    truth_m = [1 3 4 1 2 1 2 4
               1 2 4 1 3 2 2 3
               5 2 3 3 4 5 2 1]
    truth_data = DataAccessor(truth_m)

    # average error should be less than 2.5 in the 1-to-5 rating
    @test evaluate(recommender, truth_data, RMSE()) < 2.5
end

function test_evaluate_implicit()
    println("-- Testing evaluate function for implicit feedback")

    m = [NaN 1 NaN 0 0 0 NaN 1
         0 0 NaN NaN 1 0 NaN 1
         NaN 0 1 1 NaN 1 NaN 0]
    data = DataAccessor(m)
    recommender = MF(data, 2)
    build!(recommender)

    truth_m = [0 1 1 0 0 0 0 1
               0 0 1 0 1 0 0 1
               1 0 1 1 1 1 0 0]
    truth_data = DataAccessor(truth_m)

    # average error should be less than or equal to 0.5 in the [0, 1] scale
    @test evaluate(recommender, truth_data, Recall(), 4) <= 0.5
end

test_evaluate_explicit()
test_evaluate_implicit()

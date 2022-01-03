function test_evaluate_explicit()
    println("-- Testing evaluate function for explicit feedback")

    m = [missing 3 missing 1 2 1 missing 4
         1 2 missing missing 3 2 missing 3
         missing 2 3 3 missing 5 missing 1]
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

    m = [missing 1 missing 0 0 0 missing 1
         0 0 missing missing 1 0 missing 1
         missing 0 1 1 missing 1 missing 0]
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

function test_evaluate_explicit(v)
    m = [v 3 v 1 2 1 v 4
         1 2 v v 3 2 v 3
         v 2 3 3 v 5 v 1]
    data = DataAccessor(ismissing(v) ? m : sparse(m))
    recommender = MF(data, 2)
    build!(recommender)

    truth_m = [1 3 4 1 2 1 2 4
               1 2 4 1 3 2 2 3
               5 2 3 3 4 5 2 1]
    truth_data = DataAccessor(truth_m)

    # average error should be less than 2.5 in the 1-to-5 rating
    @test evaluate(recommender, truth_data, RMSE()) < 2.5
end

function test_evaluate_implicit(v)
    m = [v 1 v 0 0 0 v 1
         0 0 v v 1 0 v 1
         v 0 1 1 v 1 v 0]
    data = DataAccessor(ismissing(v) ? m : sparse(m))
    recommender = MF(data, 2)
    build!(recommender)

    truth_m = [0 1 1 0 0 0 0 1
               0 0 1 0 1 0 0 1
               1 0 1 1 1 1 0 0]
    truth_data = DataAccessor(truth_m)

    # average error should be less than or equal to 0.5 in the [0, 1] scale
    @test evaluate(recommender, truth_data, Recall(), 4) <= 0.5
end

println("-- Testing evaluate function for explicit feedback")
test_evaluate_explicit(missing)
test_evaluate_explicit(0)

println("-- Testing evaluate function for implicit feedback")
test_evaluate_implicit(missing)
test_evaluate_implicit(0)

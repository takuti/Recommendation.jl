function test_metrics_type_validation()
    metrics = [MAE(), Recall()]
    check_metrics_type(metrics, Union{AccuracyMetric, RankingMetric})
    @test_throws ErrorException check_metrics_type(metrics, Union{RankingMetric, AggregatedMetric})
end

function test_evaluate_explicit(v)
    m = [v 3 v 1 2 1 v 4
         1 2 v v 3 2 v 3
         v 2 3 3 v 5 v 1]
    data = DataAccessor(isa(v, Unknown) ? m : sparse(m))
    recommender = MF(data, 2)
    fit!(recommender)

    truth_m = [1 3 4 1 2 1 2 4
               1 2 4 1 3 2 2 3
               5 2 3 3 4 5 2 1]
    truth_data = DataAccessor(truth_m)

    # average error should be less than 2.5 in the 1-to-5 rating
    @test 0.0 < evaluate(recommender, truth_data, RMSE()) < 2.5
end

function test_evaluate_implicit(v)
    m = [v 1 v 0 0 0 v 1
         0 0 v v 1 0 v 1
         v 0 1 1 v 1 v 0]
    data = DataAccessor(isa(v, Unknown) ? m : sparse(m))
    recommender = MF(data, 2)
    fit!(recommender)

    truth_m = [0 1 1 0 0 0 0 1
               0 0 1 0 1 0 0 1
               1 0 1 1 1 1 0 0]
    truth_data = DataAccessor(truth_m)

    # must return a meaningful non-zero value as an error, but it shouldn't be too good (>0.8)
    @test 0.0 < evaluate(recommender, truth_data, Recall(), 4) <= 0.8
end

println("-- Testing metrics type validation function")
test_metrics_type_validation()

println("-- Testing evaluate function for explicit feedback")
test_evaluate_explicit(nothing)
test_evaluate_explicit(0)

println("-- Testing evaluate function for implicit feedback")
test_evaluate_implicit(nothing)
test_evaluate_implicit(0)

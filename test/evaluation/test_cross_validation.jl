function test_cross_validation_accuracy(v)
    m = [v 3 v 1 2 1 v 4
         1 2 v v 3 2 v 3
         v v v v v v v v
         v 2 3 3 v 5 v 1]
    data = DataAccessor(isa(v, Unknown) ? m : sparse(m))

    fold = 5

    # MF(data, 2)
    @test cross_validation(fold, MAE, MF, data, 2) <= 2.5

    # UserMean(data)
    @test cross_validation(fold, MAE, UserMean, data) <= 2.5

    # ItemMean(data)
    @test cross_validation(fold, MAE, ItemMean, data) <= 2.5

    # UserKNN(data, 2, true)
    @test cross_validation(fold, MAE, UserKNN, data, 2, true) <= 2.5
end

function test_cross_validation_ranking(v)
    m = [v 3 v 1 2 1 v 4
         1 2 v v 3 2 v 3
         v 2 3 3 v 5 v 1]
    data = DataAccessor(isa(v, Unknown) ? m : sparse(m))

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

println("-- Testing cross validation with accuracy metrics")
test_cross_validation_accuracy(nothing)
test_cross_validation_accuracy(0)

println("-- Testing cross validation with ranking metrics")
test_cross_validation_ranking(nothing)
test_cross_validation_ranking(0)

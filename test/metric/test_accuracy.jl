function test_rmse(truth::AbstractVector, prediction::AbstractVector)
    @test measure(RMSE(), truth, prediction) == 3.0
end

function test_mae(truth::AbstractVector, prediction::AbstractVector)
    @test measure(MAE(), truth, prediction) == 3.0
end

println("-- Testing accuracy metrics")

truth = [1, 2, 3]
prediction = [4, 5, 6]

test_rmse(truth, prediction)
test_mae(truth, prediction)

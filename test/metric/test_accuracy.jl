function test_rmse(truth::AbstractVector, prediction::AbstractVector, expected::AbstractFloat)
    @test measure(RMSE(), truth, prediction) == expected
end

function test_mae(truth::AbstractVector, prediction::AbstractVector, expected::AbstractFloat)
    @test measure(MAE(), truth, prediction) == expected
end

println("-- Testing accuracy metrics")

test_rmse([1, 2, 3], [4, 5, 6], 3.0)
test_mae([1, 2, 3], [4, 5, 6], 3.0)

test_rmse([], [], 0.0)
test_mae([], [], 0.0)

@test_throws ErrorException measure(RMSE(), [1], [])
@test_throws ErrorException measure(MAE(), [], [1])

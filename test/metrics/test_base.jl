
println("-- Testing base module for metrics calculation")

@test count_true_positive([1, 2, 3], [1, 2, 3, 4]) == 3
@test count_true_positive(Array{Int, 1}(), [1, 2, 3, 4]) == 0
@test count_true_positive([1, 2, 3], Array{Int, 1}()) == 0
@test count_true_positive(["a", "b", "c"], ["a", "b", "c"]) == 3

@test coverage([1, 2, 3], [1, 2, 3, 4]) == 0.75
@test coverage(Set(["a", "b", "c", "d"]), Set(["c", "d", "e", "f"])) == 0.5


println("-- Testing base module for metrics calculation")

@test coverage([1, 2, 3], [1, 2, 3, 4]) == 0.75
@test coverage(Set(["a", "b", "c", "d"]), Set(["c", "d", "e", "f"])) == 0.5

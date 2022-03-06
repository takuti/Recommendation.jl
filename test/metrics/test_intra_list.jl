println("-- Testing intra-list evaluation metrics")

@test count_intersect([1, 2, 3], [1, 2, 3, 4]) == 3
@test count_intersect(Array{Int, 1}(), [1, 2, 3, 4]) == 0
@test count_intersect([1, 2, 3], Array{Int, 1}()) == 0
@test count_intersect(["a", "b", "c"], ["a", "b", "c"]) == 3

@test measure(Coverage(), [1, 2, 3], catalog=[1, 2, 3, 4]) == 0.75
@test measure(Coverage(), Set(["a", "b", "c", "d"]), catalog=Set(["c", "d", "e", "f"])) == 0.5

@test measure(Novelty(), [1, 2, 3], observed=[2]) == 2  # novel items: {1, 3}

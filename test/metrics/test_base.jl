
println("-- Testing base module for metrics calculation")

@test count_intersect([1, 2, 3], [1, 2, 3, 4]) == 3
@test count_intersect(Array{Int, 1}(), [1, 2, 3, 4]) == 0
@test count_intersect([1, 2, 3], Array{Int, 1}()) == 0
@test count_intersect(["a", "b", "c"], ["a", "b", "c"]) == 3

recommendations = [[1, 2, 3], [2, 4, 6, 8], [1, 5, 10]]

@test find_all_items(recommendations) == Set([1, 2, 3, 4, 5, 6, 8, 10])
@test aggregated_diversity(recommendations) == 8
@test novelty(recommendations, [[1, 2, 3], [1, 2, 3, 5], [2]]) == 2.0  # (0 + 3 + 3) / 3

@test count_users_contain(2, recommendations) == 2
@test entropy([[1]], 1) == 0.0  # - (1 / 1) * log(1 / 1)
@test gini([[1]], 1) == 1.0  # 2 * ((1 + 1 - 1) / (1 + 1)) * 1 = 2 * 1 / 2 = 1

@test coverage([1, 2, 3], [1, 2, 3, 4]) == 0.75
@test coverage(Set(["a", "b", "c", "d"]), Set(["c", "d", "e", "f"])) == 0.5

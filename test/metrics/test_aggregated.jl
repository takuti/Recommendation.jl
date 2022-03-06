
println("-- Testing inter-list aggregated evaluation metrics")

recommendations = [[1, 2, 3], [2, 4, 6, 8], [1, 5, 10]]

@test find_all_items(recommendations) == Set([1, 2, 3, 4, 5, 6, 8, 10])
@test measure(AggregatedDiversity(), recommendations) == 8

@test count_users_contain(2, recommendations) == 2
@test measure(ShannonEntropy(), [[1]], k=1) == 0.0  # - (1 / 1) * log(1 / 1)
@test measure(GiniIndex(), [[1]], k=1) == 0.0
@test measure(GiniIndex(), [[1, 2], [1, 3]], k=2) == 0.25  # [(2*1-3-1) * 1/4 + (2*2-3-1) * 1/4 + (2*3-3-1) * 2/4] / [3-1] = 0.25

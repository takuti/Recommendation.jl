function test_factorization_machines()
    println("-- Testing Factorization Machines-based recommender")

    m = [NaN 3 NaN 1 2 1 NaN 4
         1 2 NaN NaN 3 2 NaN 3
         NaN 2 3 3 NaN 5 NaN 1]
    data = DataAccessor(m)

    num_factors = 2
    learning_rate = 15e-4
    max_iter = 100

    user = 1
    topk = 4
    items = [i for i in 1:8]

    mf = MatrixFactorization(data, num_factors)
    build!(mf, learning_rate=learning_rate, max_iter=max_iter)
    mf_rec = recommend(mf, user, topk, items)

    fm = FactorizationMachines(data, num_factors)
    build!(fm, learning_rate=learning_rate, max_iter=max_iter)
    fm_rec = recommend(fm, user, topk, items)

    # if no contextual features are considered in FMs, the recommendation
    # result must be same as matrix factorization.
    @test [item for (item, score) in mf_rec] == [item for (item, score) in fm_rec]
end

function test_factorization_machines_with_attributes()
    println("-- Testing Factorization Machines-based recommender with contextual user/item attributes")

    m = [NaN 3 NaN 1 2 1 NaN 4
         1 2 NaN NaN 3 2 NaN 3
         NaN 2 3 3 NaN 5 NaN 1]
    data = DataAccessor(m)

    set_user_attribute(data, 1, [1, 0, 20])
    set_user_attribute(data, 2, [0, 1, 23])
    set_user_attribute(data, 3, [0, 1, 12])

    set_item_attribute(data, 1, [1, 1, 0])
    set_item_attribute(data, 2, [0, 1, 1])
    set_item_attribute(data, 3, [1, 1, 1])
    set_item_attribute(data, 4, [1, 0, 0])
    set_item_attribute(data, 5, [1, 0, 1])
    set_item_attribute(data, 6, [0, 0, 1])
    set_item_attribute(data, 7, [0, 1, 0])
    set_item_attribute(data, 8, [1, 1, 1])

    recommender = FactorizationMachines(data, 2)
    build!(recommender, learning_rate=15e-4, max_iter=100)

    rec = recommend(recommender, 1, 4, [i for i in 1:8])
    @test size(rec, 1) == 4  # top-4 recos
end

test_factorization_machines()
test_factorization_machines_with_attributes()

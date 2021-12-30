function test_factorization_machines()
    println("-- Testing Factorization Machines-based feature recommender")

    m = [NaN 3 NaN 1 2 1 NaN 4
         1 2 NaN NaN 3 2 NaN 3
         NaN 2 3 3 NaN 5 NaN 1]
    data = DataAccessor(m)

    users = [1 0 20
             0 1 23
             0 1 12]
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
    build!(recommender, learning_rate=3e-8, max_iter=100)

    rec = recommend(recommender, 1, 4, [i for i in 1:8])
    @test size(rec, 1) == 4  # top-4 recos
end

test_factorization_machines()

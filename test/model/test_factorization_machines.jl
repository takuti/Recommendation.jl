function test_factorization_machines(v)
    m = [v 3 v 1 2 1 v 4
         1 2 v v 3 2 v 3
         v 2 3 3 v 5 v 1]
    data = DataAccessor(isa(v, Unknown) ? m : sparse(m))

    num_factors = 2
    learning_rate = 0.3
    max_iter = 100

    user = 1
    topk = 4
    items = [i for i in 1:8]

    mf = MatrixFactorization(data, num_factors)
    fit!(mf, learning_rate=learning_rate, max_iter=max_iter)
    mf_rec = recommend(mf, user, topk, items)

    fm = FactorizationMachines(data, num_factors)
    fit!(fm, learning_rate=learning_rate, max_iter=max_iter)
    fm_rec = recommend(fm, user, topk, items)

    # if no contextual features are considered in FMs, the recommendation
    # result must be same as matrix factorization.
    @test [item for (item, score) in mf_rec] == [item for (item, score) in fm_rec]
end

function test_factorization_machines_with_attributes(v)
    m = [v 3 v 1 2 1 v 4
         1 2 v v 3 2 v 3
         v 2 3 3 v 5 v 1]
    data = DataAccessor(isa(v, Unknown) ? m : sparse(m))

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
    fit!(recommender, reg_w0=0.3, reg_w=0.3, reg_V=0.3, learning_rate=3e-10, random_init=true)

    rec = recommend(recommender, 1, 4, [i for i in 1:8])
    @test size(rec, 1) == 4  # top-4 recos
end

println("-- Testing Factorization Machines-based recommender")
test_factorization_machines(nothing)
test_factorization_machines(0)

println("-- Testing Factorization Machines-based recommender with contextual user/item attributes")
test_factorization_machines_with_attributes(nothing)
test_factorization_machines_with_attributes(0)

function test_data_accessor()
    println("-- Testing data accessor")

    events = [Event(1, 2, 1), Event(3, 2, 1), Event(2, 6, 4)]
    n_user = 5
    n_item = 10

    data = DataAccessor(events, n_user, n_item)
    set_user_attribute(data, 1, [1, 2, 3, 4, 5])
    set_item_attribute(data, 5, [2, 4, 8, 16, 32])

    @test size(data.R) == (5, 10)
    @test data.user_attributes[1] == [1, 2, 3, 4, 5]
    @test data.item_attributes[5] == [2, 4, 8, 16, 32]

    data = DataAccessor(sparse([1 0 0; 4 5 0]))
    @test size(data.R) == (2, 3)
end

test_data_accessor()

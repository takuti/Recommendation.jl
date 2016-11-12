function test_data_accessor()
    println("-- Testing data accessor")

    events = [Event(1, 2, 1), Event(3, 2, 1), Event(2, 6, 4)]
    n_user = 5
    n_item = 10

    da = DataAccessor(events, n_user, n_item)
    set_user_attribute(da, 1, [1, 2, 3, 4, 5])
    set_item_attribute(da, 5, [2, 4, 8, 16, 32])

    @test size(da.R) == (5, 10)
    @test da.user_attributes[1] == [1, 2, 3, 4, 5]
    @test da.item_attributes[5] == [2, 4, 8, 16, 32]
end

test_data_accessor()

function test_load_movielens_100k()
    println("-- Testing to create and load a mock MovieLens 100k data file")
    path, io = mktemp()

    write(io, "1\t1\t3\n")
    write(io, "1\t1682\t2\n")
    write(io, "943\t1\t4\n")
    flush(io)

    data = load_movielens_100k(path)

    @test data.R[1, 1] == 3.0
    @test data.R[1, 1682] == 2.0
    @test data.R[943, 1] == 4.0
    @test data.R[2, 2] == 0.0
end

test_load_movielens_100k()

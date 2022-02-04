function test_load_movielens_100k()
    println("-- Testing to create and load a mock MovieLens 100k data file")
    path, io = mktemp()

    write(io, "1\t1\t3\n")
    write(io, "1\t1682\t2\n")
    write(io, "943\t1\t4\n")
    flush(io)

    R = load_movielens_100k(path)

    @test R[1, 1] == 3
    @test R[1, 1682] == 2
    @test R[943, 1] == 4
    @test ismissing(R[2, 2])
end

test_load_movielens_100k()

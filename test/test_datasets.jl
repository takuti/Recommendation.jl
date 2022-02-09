function test_get_data_home()
    dir = mktempdir()
    println("-- Testing to get a data home with a temp directory: $dir")
    ENV["JULIA_RECOMMENDATION_DATA"] = dir
    @test get_data_home() == dir
end

function test_download_file()
    # https://archive.ics.uci.edu/ml/datasets/iris
    iris_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    path = download_file(iris_url)
    @test isfile(path)
    @test_throws ErrorException download_file(iris_url, path)
end

function test_unzip()
    dir = mktempdir()
    zip_path = joinpath(dir, "example.zip")
    println("-- Testing unzip function with a dummy zip file: $zip_path")

    # https://github.com/fhs/ZipFile.jl/blob/0da33b53089f39e2a600f8b07c0e7eec2c2b9186/src/ZipFile.jl#L10-L34
    w = ZipFile.Writer(zip_path)
    f = ZipFile.addfile(w, "hello.txt")
    write(f, "hello world!\n")
    f = ZipFile.addfile(w, "julia.txt", method=ZipFile.Deflate)
    write(f, "Julia\n"^5)
    close(w)

    exdir = unzip(zip_path)
    @test exdir == dir
    @test isfile(joinpath(exdir, "hello.txt"))
end

function test_load_movielens_100k()
    path = tempname()
    println("-- Testing to download and read the MovieLens 100k data file: $path")
    data = load_movielens_100k(path)

    @test data.R[196, 242] == 3.0
    @test data.R[186, 302] == 3.0
    @test data.R[22, 377] == 1.0
    @test data.R[2, 2] == 0.0
end

test_get_data_home()
test_download_file()
test_unzip()
test_load_movielens_100k()

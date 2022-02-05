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

function test_load_movielens_100k()
    path = joinpath(tempname(), "u.data")
    println("-- Testing to download and read the MovieLens 100k data file: $path")
    data = load_movielens_100k(path)

    @test data.R[196, 242] == 3.0
    @test data.R[186, 302] == 3.0
    @test data.R[22, 377] == 1.0
    @test data.R[2, 2] == 0.0
end

test_get_data_home()
test_download_file()
test_load_movielens_100k()

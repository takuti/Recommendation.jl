using ZipFile

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
    @test path == download_file(iris_url, path)
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

function test_load_libsvm_file()
    println("-- Testing libsvm file loader")

    f = tempname()
    write(f, "1.0 0:1.0 2:-1.0\n-1.0 1:10 3:-10\n")

    X, y = load_libsvm_file(f, zero_based=true)

    @test size(X) == (2, 4)
    @test X == [1.0 0.0 -1.0 0.0; 0.0 10.0 0.0 -10.0]
    @test length(y) == 2
    @test y == [1.0, -1.0]
end

function test_load_movielens_100k()
    dir = mktempdir()
    println("-- Testing to download and read the MovieLens 100k data without specifying a path (JULIA_RECOMMENDATION_DATA=$dir)")
    ENV["JULIA_RECOMMENDATION_DATA"] = dir
    data = load_movielens_100k()
    validate_movielens_100k(data)

    path = tempname()
    println("-- Testing to download and read the MovieLens 100k data file: $path")
    data = load_movielens_100k(path)
    validate_movielens_100k(data)
end

function validate_movielens_100k(data::DataAccessor)
    @test data.R[196, 242] == 3.0
    @test data.R[186, 302] == 3.0
    @test data.R[22, 377] == 1.0
    @test data.R[2, 2] == 0.0

    # 1|24|M|technician|85711 in `u.user`
    # 21 occupations in total, and "technician" is on the 20th line of `u.occupation`
    expected_user_attribute = [
        24,
        1, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0
    ]
    @test get_user_attribute(data, 1) == expected_user_attribute

    # Genres from 1st row of `u.item` - Toy Story
    expected_item_attribute = [
        0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    ]
    @test get_item_attribute(data, 1) == expected_item_attribute
end

function test_load_movielens_latest()
    path = tempname()
    println("-- Testing to download and read the MovieLens latest (small) data file: $path")
    data = load_movielens_latest(path)

    @test data.R[1, 1] == 4.0
    @test data.R[1, 2] == 4.0
    @test data.R[1, 3] == 4.0
    @test data.R[1, 4] == 5.0

    # Genres from 1st item in `movies.csv` - Toy Story: Adventure|Animation|Children|Comedy|Fantasy
    expected_item_attribute = [
        0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0
    ]
    @test get_item_attribute(data, 1) == expected_item_attribute
end

function test_load_amazon_review()
    println("-- Testing download and read Amazon Review Dataset")
    @test_throws ErrorException load_amazon_review(category="foo")

    data = load_amazon_review(category="Magazine_Subscriptions") # smallest category
    @test data.R[1, 1] == 5.0
end

function test_load_lastfm()
    path = tempname()
    println("-- Testing download and read Last.FM user-artist listening frequency dataset at: $path")

    data = load_lastfm(path)
    @test data.R[1, 1] == 13883
end

test_get_data_home()
test_unzip()
test_load_libsvm_file()

@testset_if "download" begin
    test_download_file()
    test_load_movielens_100k()
    test_load_movielens_latest()
    test_load_amazon_review()
    test_load_lastfm()
end

export Recommender
export predict

abstract Recommender

function predict(recommender::Recommender, u::Int, i::Int)
    # noop
end

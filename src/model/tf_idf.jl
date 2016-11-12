export TFIDF

immutable TFIDF <: Recommender
    da::DataAccessor # document x attribute
    tf::AbstractMatrix # 1 x attribute
    idf::AbstractMatrix # 1 x attribute
end

function predict(recommender::TFIDF, u::Int, i::Int)
    uv = recommender.da.user_attributes[u]
    profile = recommender.da.R' * uv # attribute x 1
    ((recommender.da.R .* recommender.idf) * profile)[i]
end

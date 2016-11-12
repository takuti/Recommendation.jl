export TFIDF

immutable TFIDF <: ContentRecommender
    da::DataAccessor # document x attribute
    tf::AbstractMatrix # 1 x attribute
    idf::AbstractMatrix # 1 x attribute
end

function predict(recommender::TFIDF, uv::AbstractVector, i::Int)
    profile = recommender.da.R' * uv # attribute x 1
    ((recommender.da.R .* recommender.idf) * profile)[i]
end

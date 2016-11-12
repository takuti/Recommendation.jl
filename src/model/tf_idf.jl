export TFIDF

immutable TFIDF <: Recommender
    da::DataAccessor # document x attribute
    tf::AbstractMatrix # 1 x attribute
    idf::AbstractMatrix # 1 x attribute
end

function predict(rec::TFIDF, u::Int, i::Int)
    uv = rec.da.user_attributes[u]
    profile = rec.da.R' * uv # attribute x 1
    ((rec.da.R .* rec.idf) * profile)[i]
end

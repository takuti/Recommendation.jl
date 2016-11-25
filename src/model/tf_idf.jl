export TFIDF

immutable TFIDF <: Recommender
    da::DataAccessor # document x attribute
    tf::AbstractMatrix # 1 x attribute
    idf::AbstractMatrix # 1 x attribute
    states::States
end

TFIDF(da::DataAccessor, tf::AbstractMatrix, idf::AbstractMatrix) = begin
    # instanciate with dummy status (i.e., always true)
    TFIDF(da, tf, idf, States(:is_built => true))
end

function predict(rec::TFIDF, u::Int, i::Int)
    check_build_status(rec)

    uv = rec.da.user_attributes[u]
    profile = rec.da.R' * uv # attribute x 1
    ((rec.da.R .* rec.idf) * profile)[i]
end

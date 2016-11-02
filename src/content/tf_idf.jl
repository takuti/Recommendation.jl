export TFIDF

immutable TFIDF <: ContentRecommender
    m::SparseMatrixCSC # document x attribute
    tf::Array{Int,2} # 1 x attribute
    idf::Array{Float64,2} # 1 x attribute
end

TFIDF(m::SparseMatrixCSC; does_use_idf::Bool=false, is_normalized::Bool=false) = begin
    tf = sum(m, 1)

    idf = ones(size(tf))
    if does_use_idf
        idf = 1 ./ tf
    end

    if is_normalized
        m = m ./ sqrt(sum(m.^2, 2))
    end

    TFIDF(m, tf, idf)
end

function predict(recommender::TFIDF, uv::SparseVector, i::Int)
    profile = recommender.m' * uv # attribute x 1
    ((recommender.m .* recommender.idf) * profile)[i]
end


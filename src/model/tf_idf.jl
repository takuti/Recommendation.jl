export TFIDF

"""
    TFIDF(
        data::DataAccessor,
        tf::AbstractMatrix,
        idf::AbstractMatrix
    )

Content-based recommendation using TF-IDF scoring. TF and IDF matrix are respectively specified as `tf` and `idf`.

More concretely, each item is represented as a set of words, and the items are modeled by TF-IDF weighting of the words. The technique was initially used in information retrieval to model a document-word matrix. In case of our item-word matrices, for a given item ``i``, term frequency (TF) for a term ``t`` is defined as:

```math
\\mathrm{tf}(t, i) = \\frac{n_{t,i}}{N_i},
```

where ``n_{t,i}`` denotes an ``(i, t)`` element in ``I``, and ``N_i`` is the total number of words that an item ``i`` contains. So, ``\\mathrm{tf}(t, i)`` characterizes an item-term pair by normalized frequency of terms. Meanwhile, inverse document frequency (IDF) is computed over ``M`` items as:

```math
\\mathrm{idf}(t) = \\log \\frac{M}{\\mathrm{df}(t)} + 1,
```

where ``\\mathrm{df}(t)`` counts the number of items which associate with a term ``t``. What ``\\mathrm{idf}(t)`` does is to decrease the effect of commonly used terms, because general words cannot characterize a specific item.

Finally, each item-term pair is weighted by ``\\mathrm{tf}(t, i) \\cdot \\mathrm{idf}(t)`` in the TF-IDF scheme. For instance, if we like to recommend web pages to users, we first need to parse sentences on a page and then construct a vector based on the frequency of each term as follows. Similarly, in case that a recommender is running on a folksonomic (i.e. social tagging) service, the frequency of tags corresponds to each dimension of a vector, and ``\\mathrm{tf}(t, i)`` and ``\\mathrm{idf}(t)`` are finally calculated for each item-tag pair.

![tfidf](./assets/images/tfidf.png)
"""
struct TFIDF <: Recommender
    data::DataAccessor # document x attribute
    tf::AbstractMatrix # 1 x attribute
    idf::AbstractMatrix # 1 x attribute

    function TFIDF(data::DataAccessor, tf::AbstractMatrix, idf::AbstractMatrix)
        # instanciate with dummy status (i.e., always true)
        new(data, tf, idf)
    end
end

function predict(recommender::TFIDF, u::Int, i::Int)
    uv = recommender.data.user_attributes[u]
    profile = recommender.data.R' * uv # attribute x 1
    ((recommender.data.R .* recommender.idf) * profile)[i]
end

export MatrixUtils

module MatrixUtils

function pearson_correlation(mat::AbstractMatrix, axis::Int=1)
    m = (axis == 1) ? copy(mat) : mat'

    n_row = size(m)[1]
    corr = zeros(n_row, n_row)

    for ri in 1:n_row
        for rj in ri:n_row
            # pairwise correlation (i.e., ignore NaNs)
            ij = !isnan(m[ri, :]) & !isnan(m[rj, :])

            vi = m[ri, :] - mean(m[ri, ij])
            vj = m[rj, :] - mean(m[rj, ij])

            numer = dot(vi[ij], vj[ij])
            denom = sqrt(dot(vi[ij], vi[ij]) * dot(vj[ij], vj[ij]))

            c = numer / denom
            corr[ri, rj] = c
            if (ri != rj); corr[rj, ri] = c; end # symmetric
        end
    end

    corr
end

function cosine_similarity(mat::AbstractMatrix, axis::Int=1, is_adjusted::Bool=false)
    m = (axis == 1) ? mat' : copy(mat)

    n_row, n_col = size(m)
    sim = zeros(n_col, n_col)

    if is_adjusted
        # subtract mean
        for ri in 1:n_row
            indices = !isnan(m[ri, :])
            vmean = mean(m[ri, indices])
            m[ri, indices] -= vmean
        end
    end

    # unlike pearson correlation, matrix can be filled by zeros for cosine similarity
    m[isnan(m)] = 0

    # compute L2 nrom of each column
    norms = sqrt(sum(m.^2, 1))

    for ci in 1:n_col
        for cj in ci:n_col
            numer = dot(m[:, ci], m[:, cj])
            denom = norms[ci] * norms[cj]
            s = numer / denom

            sim[ci, cj] = s
            if (ci != cj); sim[cj, ci] = s; end
        end
    end

    sim
end

end # module MatrixUtils

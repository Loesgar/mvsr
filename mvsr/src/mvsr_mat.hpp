#ifndef MVSR_MAT_HPP
#define MVSR_MAT_HPP

#include <cstddef>

/**
 * @brief Computes the grand sum of $aa^T*b$ where '*' is the shor product (element-wise product)
 *
 * @param width width of matrix a
 * @param height height of matrix a
 * @param a input buffer for matrix a
 * @param b input buffer for matrix b
 * @return grand sum
 */
template <typename Scalar>
inline Scalar MatGsAAtPmulB(size_t width, size_t height, const Scalar *a, const Scalar *b)
{
    Scalar res = Scalar(0);

    const Scalar *c = a;
    for (size_t x = 0; x < height; x++, c += width)
    {
        const Scalar *d = a;
        Scalar tmp = Scalar(0);
        for (size_t y = 0; y <= x; y++, d += width)
        {
            tmp = Scalar(0);
            for (size_t i = 0; i < width; i++)
            {
                tmp += c[i] * d[i];
            }
            res += Scalar(2) * tmp * b[y * height + x];
        }
        res -= tmp * b[x * height + x];
    }

    return res;
}

/**
 * @brief Solves a linear equation as if calculating $a^{-1} * (b)$.
 *
 * @param size height of a and b, width of a
 * @param widthb width of b
 * @param res output buffer
 * @param a input buffer for matrix a
 * @param b input buffer for matrix b
 */
template <typename Scalar>
inline void MatSolve(size_t size, size_t widthb, Scalar *res, const Scalar *a, const Scalar *b)
{
    if (size == 2) // use multiplication by fast inverse of 2x2 matrix
    {
        Scalar adj = Scalar(1) / (a[0] * a[3] - a[1] * a[2]);
        Scalar t1 = (adj * a[3]), t2 = (adj * -a[1]);
        Scalar t3 = (adj * -a[2]), t4 = (adj * a[0]);
        for (size_t i = 0; i < widthb; i++)
        {
            Scalar res1 = t1 * b[i] + t2 * b[widthb + i];
            Scalar res2 = t3 * b[i] + t4 * b[widthb + i];
            res[i] = res1;
            res[i + widthb] = res2;
        }
    }
    else if (size == 1) // use inverse multiplication
    {
        Scalar inv = Scalar(1) / a[0];
        for (size_t i = 0; i < widthb; i++)
        {
            res[i] = b[i] * inv;
        }
    }
    else // use generic gauss-jordan
    {
        // Copy to big matrix
        const size_t RowSize = size + widthb;
        Scalar mat[size * RowSize];

        Scalar *curRow = mat;
        for (size_t y = 0; y < size; y++)
        {
            for (size_t i = 0; i < size; i++)
            {
                *curRow++ = *a++;
            }
            for (size_t i = 0; i < widthb; i++)
            {
                *curRow++ = *b++;
            }
        }

        // gauss-jordan (down)
        curRow = mat;
        for (size_t y = 0; y < size; y++)
        {
            Scalar *subRow = mat;
            for (size_t x = 0; x < y; x++)
            {
                const Scalar mul = -curRow[x];
                for (size_t i = 0; i < RowSize; i++)
                {
                    curRow[i] += mul * subRow[i];
                }
                subRow += RowSize;
            }

            const Scalar div = mat[y * RowSize + y];
            for (size_t x = y; x < RowSize; x++)
            {
                curRow[x] /= div;
            }
            curRow += RowSize;
        }

        // gauss-jordan (up)
        curRow -= RowSize;
        for (size_t x = size; x-- != 0; curRow -= RowSize)
        {
            Scalar *otherRow = curRow - RowSize;
            for (size_t y = x; y-- != 0; otherRow -= RowSize)
            {
                const Scalar mul = -otherRow[x];
                for (size_t i = x; i < RowSize; i++)
                {
                    otherRow[i] += mul * curRow[i];
                }
            }
        }

        // copy back result
        for (size_t y = 0; y < size; y++)
        {
            curRow += RowSize;
            for (size_t x = size; x < RowSize; x++) *res++ = curRow[x];
        }
    }
}

#endif // guard

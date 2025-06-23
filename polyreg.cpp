#include <Eigen>
#include <algorithm>
#include <cmath>
#include <span>

#include "heap.hpp"
#include "list.hpp"

/* Above this threshold the Sherman–Morrison formula will be used for a rank one update in the optimize step. */
#ifndef PWREG_SM_THRESHOLD
#define PWREG_SM_THRESHOLD 4
#endif

/* This class represents a single segment, i.e. one continous OLS in the way described in the paper.  */
template<size_t Dimensions, typename Scalar>
class Part
{
public:
    using Mat = Eigen::Matrix<Scalar, Dimensions, Dimensions>;
    using Vec = Eigen::Matrix<Scalar, Dimensions, 1>;

    template <typename Range>
    explicit Part(const Range &r)
    {
        xvals.setZero();
        yvals.setZero();
        y2 = Scalar(0);
        size = 0;

        for (auto [xVals, y] : r)
        {
            const Eigen::Map<const Vec> mat(xVals);
            // Vec mat;
            // std::ranges::copy(std::span<Scalar, Dimensions>(xVals), mat.begin());
            xvals += mat * mat.transpose();
            yvals += y * mat;
            y2 += y*y;
            size++;
        }

        error = (size > Dimensions) ? NAN : Scalar(0);
    }

    template <typename Range>
    void removePoints(const Range &r)
    {
        for (auto [xVals, y] : r)
        {
            const Eigen::Map<const Vec> mat(xVals);
            // Vec mat;
            // std::ranges::copy(std::span<Scalar, Dimensions>(xVals), mat.begin());
            xvals -= mat * mat.transpose();
            yvals -= y * mat;
            y2 -= y*y;
            size--;
        }

        error = (size > Dimensions) ? NAN : Scalar(0);
    }

    template <typename Range>
    void addPoints(const Range &r)
    {
        for (auto [xVals, y] : r)
        {
            const Eigen::Map<const Vec> mat(xVals);
            // Vec mat;
            // std::ranges::copy(std::span<Scalar, Dimensions>(xVals), mat.begin());
            xvals += mat * mat.transpose();
            yvals += y * mat;
            y2 += y*y;
            size++;
        }

        error = (size > Dimensions) ? NAN : Scalar(0);
    }

    Part merge(const Part &successor) const
    {
        Part res;
        res.xvals = xvals + successor.xvals;
        res.yvals = yvals + successor.yvals;
        res.y2 = y2 + successor.y2;
        res.size = size + successor.size;
        res.error = (res.size > Dimensions) ? NAN : Scalar(0);
        return res;
    }

    Vec getModel()
    {
        Vec model;
        if (size >= Dimensions)
        {
            model = xvals.ldlt().solve(yvals);
            //model = xvals.partialPivLu().solve(yvals);
            //model = xvals.inverse() * yvals;
        }
        else
        {
            auto tmp = xvals;
            for (auto cur = size; cur < Dimensions; cur++)
            {
                for (auto i = 0; i < cur; i++)
                    tmp(cur,i) = tmp(i,cur) = Scalar(0);
                tmp(cur,cur) = Scalar(1);
            }
            model = tmp.ldlt().solve(yvals);
            //model = tmp.partialPivLu().solve(yvals);
            //model = tmp.inverse() * yvals;
            for (auto i = size; i < Dimensions; i++)
                model(i) = Scalar(0);
        }
        return model;
    }
    Scalar getError(const Vec &model) const
    {
        Scalar e = (model * model.transpose()).cwiseProduct(xvals).sum();
        e -= Scalar(2) * yvals.array().cwiseProduct(model.array()).sum();
        e += y2;
        return e;
    }
    Scalar getError()
    {
        if (std::isnan(error))
        {
            error = getError(getModel());
        }
        return error;
    }

    size_t getSize() const
    {
        return size;
    }

    static Part Zero()
    {
        Part res;
        res.xvals.setZero();
        res.yvals.setZero();
        return res;
    }

    Mat getXVals() const
    {
        return xvals;
    }
    Vec getYVals() const
    {
        return yvals;
    }
    Scalar getY2() const
    {
        return y2;
    }

private:
    Part() = default;
    Mat xvals;          // Matrix A from the paper
    Vec yvals;          // Matrix B from the paper
    Scalar y2 = 0;      // Summed y^2 (rest to get to matrix C)
    Scalar error = 0;   // Cached version of current RSS
    size_t size = 0;    // Number of samples in this segment
};

/* This class represents a regression. It contians the double linked list with the segments and the priority queue. */
template <typename Scalar, size_t Dimension>
class Regression
{
public:
    using Part = Part<Dimension, Scalar>;

    Regression() = default;
    Regression(const Regression<Scalar, Dimension> &other)
    {
        pieces = other.pieces;
        queue = other.queue.copyByOrder(other.pieces, pieces);
    }
    ~Regression() = default;

    Regression &reduce(size_t numBreaks)
    {
        while (queue.size() > numBreaks)
        {
            merge(queue.pop().second);
        }
        return *this;
    }
    void prepare(std::span<Scalar[Dimension+1]> xyvals)
    {
        struct xytuple {Scalar x[Dimension]; Scalar y;};
        auto data = std::span<xytuple>{(xytuple*)xyvals.data(), xyvals.size()};

        pieces.prepend(Segment{ .curSegment = Part(data.subspan(data.size() - Dimension - (data.size() % Dimension))) });
        queue.reserve((data.size() / Dimension) - 1);
        auto last = pieces.begin();

        for (data = data.subspan(0, data.size() - Dimension - (data.size() % Dimension)); !data.empty(); data = data.subspan(0, data.size() - Dimension))
        {
            auto it = pieces.prepend(Segment { .curSegment = Part(data.subspan(data.size() - Dimension)) });
            queue.pushProvisionally(last->curSegment.merge(it->curSegment).getError() - (last->curSegment.getError() + it->curSegment.getError()), *it);
            last = it;
        }

        queue.heapify();
    }
    auto get()
    {
        return pieces;
    }

    size_t optimize(std::span<Scalar[Dimension+1]> xyvals)
    {
        struct xytuple {Scalar x[Dimension]; Scalar y;};
        auto data = std::span<xytuple>{(xytuple*)xyvals.data(), xyvals.size()};

        const auto optimizeBp = [&data](Part &s1, Part &s2, size_t bpIdx)
        {
            auto oldlen = s1.getSize() + s2.getSize();
            const size_t start = bpIdx - s1.getSize() / 4;
            const size_t end = bpIdx + s2.getSize() / 4;
            s1.removePoints(data.subspan(start, bpIdx - start));
            s2.addPoints(data.subspan(start, bpIdx - start));

            size_t idx = start;
            Scalar err;
            typename Part::Mat x1, x2;
            if constexpr (Dimension <= PWREG_SM_THRESHOLD)
            {
                err = s1.getError() + s2.getError();
            }
            else
            {
                x1 = s1.getXVals().inverse();
                x2 = s2.getXVals().inverse();
                err = s1.getError(x1 * s1.getYVals()) + s2.getError(x2 * s2.getYVals());
            }
            for (size_t i = start; i < end; i++)
            {
                s1.addPoints(data.subspan(i,1));
                s2.removePoints(data.subspan(i,1));

                Scalar curErr;
                if constexpr (Dimension <= PWREG_SM_THRESHOLD)
                {
                    curErr = s1.getError() + s2.getError();
                }
                else
                {
                    const Eigen::Map<const typename Part::Vec> vec(data[i].x);
                    //typename Part::Vec vec;
                    //std::ranges::copy(data[i].x, vec.begin());
                    x1 -= (x1 * vec) * (vec.transpose() * x1) / (1 + vec.transpose() * x1 * vec);
                    x2 -= (x2 * vec) * (-vec.transpose() * x2) / (1 + -vec.transpose() * x2 * vec);
                    curErr = s1.getError(x1 * s1.getYVals()) + s2.getError(x2 * s2.getYVals());
                }

                if (curErr < err)
                {
                    err = curErr;
                    idx = i+1;
                }
            }
            s1.removePoints(data.subspan(idx, end - idx));
            s2.addPoints(data.subspan(idx, end - idx));
        };

        struct OptEntry
        {
            size_t size, startPos;
            bool operator<(const OptEntry &other) const { return size > other.size; }
        };
        Heap<OptEntry, Segment> elements(queue.size() + 1);
        {
            auto start = pieces.begin();
            auto end = pieces.end();

            for (size_t startPos = start++->curSegment.getSize(); start != end; ++start)
            {
                elements.pushProvisionally({start->curSegment.getSize(), startPos}, *start);
                startPos += start->curSegment.getSize();
            }
            elements.heapify();
        }

        while (!elements.isEmpty())
        {
            auto &&[entry, seg] = elements.pop();
            size_t startPos = entry.startPos;
            size_t endPos = entry.startPos + seg.curSegment.getSize();
            auto piece = List<Segment>::Iterator::FromElement(seg);
            auto n = std::next(piece);
            auto p = std::prev(piece);
            optimizeBp(p->curSegment, piece->curSegment, startPos);
        }

        // update heap after changes
        // for (auto it = pieces.begin(); it != pieces.end(); ++it)
        //     updateHeap(it);

        // todo: for compatibilty, remove return after
        return queue.size() + 1;
    }

private:
    struct Segment : Heap<Scalar, Segment>::Reference {
        Part curSegment;
    };

    void merge(Segment &seg)
    {
        auto it = List<Segment>::Iterator::FromElement(seg);
        auto nit = std::next(it);
        nit->curSegment = it->curSegment.merge(nit->curSegment);
        pieces.remove(it);

        update(nit);
    }

    void update(const List<Segment>::Iterator &it)
    {
        if (auto next = std::next(it); next != pieces.end())
        {
            updateHeap(it);
        }
        if (it != pieces.begin())
        {
            updateHeap(std::prev(it));
        }
    }

    void updateHeap(const List<Segment>::Iterator &seg)
    {
        auto nextSeg = std::next(seg)->curSegment;
        auto newErr = (seg->curSegment.getSize() < Dimension && nextSeg.getSize() < Dimension) ? std::numeric_limits<Scalar>::max() : seg->curSegment.merge(nextSeg).getError() - (seg->curSegment.getError() + nextSeg.getError());
        queue.update(*seg, newErr);
    }

    List<Segment> pieces;           // Double linked list, containing the segments
    Heap<Scalar, Segment> queue;    // Priority queue with the merge cost values and references to the corresponding segments
};

// Initalize a regression by providing the samples. Generates n/d segments.
template <typename Scalar, size_t Dimension>
void *pwreg_init(size_t numElements, Scalar *data)
{
    auto res = new Regression<Scalar, Dimension>();
    res->prepare(std::span<Scalar[Dimension+1]>{(Scalar(*)[Dimension+1])data, numElements});
    return res;
}

// Copy a regression object.
template <typename Scalar, size_t Dimension>
void* pwreg_copy(void *regression)
{
    auto reg = (Regression<Scalar, Dimension>*)regression;
    return new Regression<Scalar, Dimension>(*reg);
}

// Delete a regression object (free memory)
template <typename Scalar, size_t Dimension>
void pwreg_delete(void *regression)
{
    auto reg = (Regression<Scalar, Dimension>*)regression;
    delete reg;
}

// Reduction step: Reduces the segment count to k (numPieces)
template <typename Scalar, size_t Dimension>
void pwreg_reduce(void *regression, size_t numPieces, size_t *breakPoints, Scalar *pieceModels, Scalar *pieceErrors)
{
    auto reg = (Regression<Scalar, Dimension>*)regression;
    reg->reduce(numPieces - 1);
    size_t res = 0;
    for (auto &p : reg->get())
    {
        *breakPoints++ = res;
        res += p.curSegment.getSize();
        pieceModels = std::ranges::copy(p.curSegment.getModel(), pieceModels).out;
        *pieceErrors++ = p.curSegment.getError();
    }
}

template <typename Scalar, size_t Dimension>
size_t pwreg_optimize(void *regression, const Scalar *data)
{
    auto reg = (Regression<Scalar, Dimension>*)regression;
    return reg->optimize(std::span<Scalar[Dimension+1]>{(Scalar(*)[Dimension+1])data, size_t(0)-1});
}

// Export functions for differnet dimension sizes and scalar types

// extern "C" void* pwreg_f32d1_init(size_t numElements, float *data)      { return pwreg_init<float, 1>(numElements, data); }
extern "C" void* pwreg_f32d2_init(size_t numElements, float *data)      { return pwreg_init<float, 2>(numElements, data); }
extern "C" void* pwreg_f32d3_init(size_t numElements, float *data)      { return pwreg_init<float, 3>(numElements, data); }
// extern "C" void* pwreg_f32d4_init(size_t numElements, float *data)      { return pwreg_init<float, 4>(numElements, data); }
// extern "C" void* pwreg_f32d5_init(size_t numElements, float *data)      { return pwreg_init<float, 5>(numElements, data); }
// extern "C" void* pwreg_f32d6_init(size_t numElements, float *data)      { return pwreg_init<float, 6>(numElements, data); }
// extern "C" void* pwreg_f32d7_init(size_t numElements, float *data)      { return pwreg_init<float, 7>(numElements, data); }
// extern "C" void* pwreg_f32d8_init(size_t numElements, float *data)      { return pwreg_init<float, 8>(numElements, data); }
// extern "C" void* pwreg_f32d9_init(size_t numElements, float *data)      { return pwreg_init<float, 9>(numElements, data); }

extern "C" void* pwreg_f64d1_init(size_t numElements, double *data)     { return pwreg_init<double, 1>(numElements, data); }
extern "C" void* pwreg_f64d2_init(size_t numElements, double *data)     { return pwreg_init<double, 2>(numElements, data); }
extern "C" void* pwreg_f64d3_init(size_t numElements, double *data)     { return pwreg_init<double, 3>(numElements, data); }
extern "C" void* pwreg_f64d4_init(size_t numElements, double *data)     { return pwreg_init<double, 4>(numElements, data); }
extern "C" void* pwreg_f64d5_init(size_t numElements, double *data)     { return pwreg_init<double, 5>(numElements, data); }
extern "C" void* pwreg_f64d6_init(size_t numElements, double *data)     { return pwreg_init<double, 6>(numElements, data); }
extern "C" void* pwreg_f64d7_init(size_t numElements, double *data)     { return pwreg_init<double, 7>(numElements, data); }
extern "C" void* pwreg_f64d8_init(size_t numElements, double *data)     { return pwreg_init<double, 8>(numElements, data); }
extern "C" void* pwreg_f64d9_init(size_t numElements, double *data)     { return pwreg_init<double, 9>(numElements, data); }
extern "C" void* pwreg_f64d16_init(size_t numElements, double *data)    { return pwreg_init<double, 16>(numElements, data); }
extern "C" void* pwreg_f64d32_init(size_t numElements, double *data)    { return pwreg_init<double, 32>(numElements, data); }
extern "C" void* pwreg_f64d64_init(size_t numElements, double *data)    { return pwreg_init<double, 64>(numElements, data); }
extern "C" void* pwreg_f64d128_init(size_t numElements, double *data)   { return pwreg_init<double, 128>(numElements, data); }
extern "C" void* pwreg_f64d256_init(size_t numElements, double *data)   { return pwreg_init<double, 256>(numElements, data); }


// extern "C" void* pwreg_f32d1_copy(void *regression)                     { return pwreg_copy<float, 1>(regression); }
extern "C" void* pwreg_f32d2_copy(void *regression)                     { return pwreg_copy<float, 2>(regression); }
extern "C" void* pwreg_f32d3_copy(void *regression)                     { return pwreg_copy<float, 3>(regression); }
// extern "C" void* pwreg_f32d4_copy(void *regression)                     { return pwreg_copy<float, 4>(regression); }
// extern "C" void* pwreg_f32d5_copy(void *regression)                     { return pwreg_copy<float, 5>(regression); }
// extern "C" void* pwreg_f32d6_copy(void *regression)                     { return pwreg_copy<float, 6>(regression); }
// extern "C" void* pwreg_f32d7_copy(void *regression)                     { return pwreg_copy<float, 7>(regression); }
// extern "C" void* pwreg_f32d8_copy(void *regression)                     { return pwreg_copy<float, 8>(regression); }
// extern "C" void* pwreg_f32d9_copy(void *regression)                     { return pwreg_copy<float, 9>(regression); }

extern "C" void* pwreg_f64d1_copy(void *regression)                     { return pwreg_copy<double, 1>(regression); }
extern "C" void* pwreg_f64d2_copy(void *regression)                     { return pwreg_copy<double, 2>(regression); }
extern "C" void* pwreg_f64d3_copy(void *regression)                     { return pwreg_copy<double, 3>(regression); }
extern "C" void* pwreg_f64d4_copy(void *regression)                     { return pwreg_copy<double, 4>(regression); }
extern "C" void* pwreg_f64d5_copy(void *regression)                     { return pwreg_copy<double, 5>(regression); }
extern "C" void* pwreg_f64d6_copy(void *regression)                     { return pwreg_copy<double, 6>(regression); }
extern "C" void* pwreg_f64d7_copy(void *regression)                     { return pwreg_copy<double, 7>(regression); }
extern "C" void* pwreg_f64d8_copy(void *regression)                     { return pwreg_copy<double, 8>(regression); }
extern "C" void* pwreg_f64d9_copy(void *regression)                     { return pwreg_copy<double, 9>(regression); }
extern "C" void* pwreg_f64d16_copy(void *regression)                    { return pwreg_copy<double, 16>(regression); }
extern "C" void* pwreg_f64d32_copy(void *regression)                    { return pwreg_copy<double, 32>(regression); }
extern "C" void* pwreg_f64d64_copy(void *regression)                    { return pwreg_copy<double, 64>(regression); }
extern "C" void* pwreg_f64d128_copy(void *regression)                   { return pwreg_copy<double, 128>(regression); }
extern "C" void* pwreg_f64d256_copy(void *regression)                   { return pwreg_copy<double, 256>(regression); }

// extern "C" void pwreg_f32d1_delete(void *regression)                    { return pwreg_delete<float, 1>(regression); }
extern "C" void pwreg_f32d2_delete(void *regression)                    { return pwreg_delete<float, 2>(regression); }
extern "C" void pwreg_f32d3_delete(void *regression)                    { return pwreg_delete<float, 3>(regression); }
// extern "C" void pwreg_f32d4_delete(void *regression)                    { return pwreg_delete<float, 4>(regression); }
// extern "C" void pwreg_f32d5_delete(void *regression)                    { return pwreg_delete<float, 5>(regression); }
// extern "C" void pwreg_f32d6_delete(void *regression)                    { return pwreg_delete<float, 6>(regression); }
// extern "C" void pwreg_f32d7_delete(void *regression)                    { return pwreg_delete<float, 7>(regression); }
// extern "C" void pwreg_f32d8_delete(void *regression)                    { return pwreg_delete<float, 8>(regression); }
// extern "C" void pwreg_f32d9_delete(void *regression)                    { return pwreg_delete<float, 9>(regression); }

extern "C" void pwreg_f64d1_delete(void *regression)                    { return pwreg_delete<double, 1>(regression); }
extern "C" void pwreg_f64d2_delete(void *regression)                    { return pwreg_delete<double, 2>(regression); }
extern "C" void pwreg_f64d3_delete(void *regression)                    { return pwreg_delete<double, 3>(regression); }
extern "C" void pwreg_f64d4_delete(void *regression)                    { return pwreg_delete<double, 4>(regression); }
extern "C" void pwreg_f64d5_delete(void *regression)                    { return pwreg_delete<double, 5>(regression); }
extern "C" void pwreg_f64d6_delete(void *regression)                    { return pwreg_delete<double, 6>(regression); }
extern "C" void pwreg_f64d7_delete(void *regression)                    { return pwreg_delete<double, 7>(regression); }
extern "C" void pwreg_f64d8_delete(void *regression)                    { return pwreg_delete<double, 8>(regression); }
extern "C" void pwreg_f64d9_delete(void *regression)                    { return pwreg_delete<double, 9>(regression); }
extern "C" void pwreg_f64d16_delete(void *regression)                   { return pwreg_delete<double, 16>(regression); }
extern "C" void pwreg_f64d32_delete(void *regression)                   { return pwreg_delete<double, 32>(regression); }
extern "C" void pwreg_f64d64_delete(void *regression)                   { return pwreg_delete<double, 64>(regression); }
extern "C" void pwreg_f64d128_delete(void *regression)                  { return pwreg_delete<double, 128>(regression); }
extern "C" void pwreg_f64d256_delete(void *regression)                  { return pwreg_delete<double, 256>(regression); }

// extern "C" void pwreg_f32d1_reduce(void *regression, size_t numPieces, size_t *breakPoints, float *pieceModels, float *pieceErrors)    { pwreg_reduce<float, 1>(regression, numPieces, breakPoints, pieceModels, pieceErrors); }
extern "C" void pwreg_f32d2_reduce(void *regression, size_t numPieces, size_t *breakPoints, float *pieceModels, float *pieceErrors)    { pwreg_reduce<float, 2>(regression, numPieces, breakPoints, pieceModels, pieceErrors); }
extern "C" void pwreg_f32d3_reduce(void *regression, size_t numPieces, size_t *breakPoints, float *pieceModels, float *pieceErrors)    { pwreg_reduce<float, 3>(regression, numPieces, breakPoints, pieceModels, pieceErrors); }
// extern "C" void pwreg_f32d4_reduce(void *regression, size_t numPieces, size_t *breakPoints, float *pieceModels, float *pieceErrors)    { pwreg_reduce<float, 4>(regression, numPieces, breakPoints, pieceModels, pieceErrors); }
// extern "C" void pwreg_f32d5_reduce(void *regression, size_t numPieces, size_t *breakPoints, float *pieceModels, float *pieceErrors)    { pwreg_reduce<float, 5>(regression, numPieces, breakPoints, pieceModels, pieceErrors); }
// extern "C" void pwreg_f32d6_reduce(void *regression, size_t numPieces, size_t *breakPoints, float *pieceModels, float *pieceErrors)    { pwreg_reduce<float, 6>(regression, numPieces, breakPoints, pieceModels, pieceErrors); }
// extern "C" void pwreg_f32d7_reduce(void *regression, size_t numPieces, size_t *breakPoints, float *pieceModels, float *pieceErrors)    { pwreg_reduce<float, 7>(regression, numPieces, breakPoints, pieceModels, pieceErrors); }
// extern "C" void pwreg_f32d8_reduce(void *regression, size_t numPieces, size_t *breakPoints, float *pieceModels, float *pieceErrors)    { pwreg_reduce<float, 8>(regression, numPieces, breakPoints, pieceModels, pieceErrors); }
// extern "C" void pwreg_f32d9_reduce(void *regression, size_t numPieces, size_t *breakPoints, float *pieceModels, float *pieceErrors)    { pwreg_reduce<float, 9>(regression, numPieces, breakPoints, pieceModels, pieceErrors); }

extern "C" void pwreg_f64d1_reduce(void *regression, size_t numPieces, size_t *breakPoints, double *pieceModels, double *pieceErrors)   { pwreg_reduce<double, 1>(regression, numPieces, breakPoints, pieceModels, pieceErrors); }
extern "C" void pwreg_f64d2_reduce(void *regression, size_t numPieces, size_t *breakPoints, double *pieceModels, double *pieceErrors)   { pwreg_reduce<double, 2>(regression, numPieces, breakPoints, pieceModels, pieceErrors); }
extern "C" void pwreg_f64d3_reduce(void *regression, size_t numPieces, size_t *breakPoints, double *pieceModels, double *pieceErrors)   { pwreg_reduce<double, 3>(regression, numPieces, breakPoints, pieceModels, pieceErrors); }
extern "C" void pwreg_f64d4_reduce(void *regression, size_t numPieces, size_t *breakPoints, double *pieceModels, double *pieceErrors)   { pwreg_reduce<double, 4>(regression, numPieces, breakPoints, pieceModels, pieceErrors); }
extern "C" void pwreg_f64d5_reduce(void *regression, size_t numPieces, size_t *breakPoints, double *pieceModels, double *pieceErrors)   { pwreg_reduce<double, 5>(regression, numPieces, breakPoints, pieceModels, pieceErrors); }
extern "C" void pwreg_f64d6_reduce(void *regression, size_t numPieces, size_t *breakPoints, double *pieceModels, double *pieceErrors)   { pwreg_reduce<double, 6>(regression, numPieces, breakPoints, pieceModels, pieceErrors); }
extern "C" void pwreg_f64d7_reduce(void *regression, size_t numPieces, size_t *breakPoints, double *pieceModels, double *pieceErrors)   { pwreg_reduce<double, 7>(regression, numPieces, breakPoints, pieceModels, pieceErrors); }
extern "C" void pwreg_f64d8_reduce(void *regression, size_t numPieces, size_t *breakPoints, double *pieceModels, double *pieceErrors)   { pwreg_reduce<double, 8>(regression, numPieces, breakPoints, pieceModels, pieceErrors); }
extern "C" void pwreg_f64d9_reduce(void *regression, size_t numPieces, size_t *breakPoints, double *pieceModels, double *pieceErrors)   { pwreg_reduce<double, 9>(regression, numPieces, breakPoints, pieceModels, pieceErrors); }
extern "C" void pwreg_f64d16_reduce(void *regression, size_t numPieces, size_t *breakPoints, double *pieceModels, double *pieceErrors)  { pwreg_reduce<double, 16>(regression, numPieces, breakPoints, pieceModels, pieceErrors); }
extern "C" void pwreg_f64d32_reduce(void *regression, size_t numPieces, size_t *breakPoints, double *pieceModels, double *pieceErrors)  { pwreg_reduce<double, 32>(regression, numPieces, breakPoints, pieceModels, pieceErrors); }
extern "C" void pwreg_f64d64_reduce(void *regression, size_t numPieces, size_t *breakPoints, double *pieceModels, double *pieceErrors)  { pwreg_reduce<double, 64>(regression, numPieces, breakPoints, pieceModels, pieceErrors); }
extern "C" void pwreg_f64d128_reduce(void *regression, size_t numPieces, size_t *breakPoints, double *pieceModels, double *pieceErrors) { pwreg_reduce<double, 128>(regression, numPieces, breakPoints, pieceModels, pieceErrors); }
extern "C" void pwreg_f64d256_reduce(void *regression, size_t numPieces, size_t *breakPoints, double *pieceModels, double *pieceErrors) { pwreg_reduce<double, 256>(regression, numPieces, breakPoints, pieceModels, pieceErrors); }

// extern "C" size_t pwreg_f32d1_optimize(void *regression, float *data)      { return pwreg_optimize<float, 1>(regression, data); }
extern "C" size_t pwreg_f32d2_optimize(void *regression, float *data)      { return pwreg_optimize<float, 2>(regression, data); }
extern "C" size_t pwreg_f32d3_optimize(void *regression, float *data)      { return pwreg_optimize<float, 3>(regression, data); }
// extern "C" void pwreg_f32d4_optimize(void *regression, float *data)      { return pwreg_optimize<float, 4>(regression, data); }
// extern "C" void pwreg_f32d5_optimize(void *regression, float *data)      { return pwreg_optimize<float, 5>(regression, data); }
// extern "C" void pwreg_f32d6_optimize(void *regression, float *data)      { return pwreg_optimize<float, 6>(regression, data); }
// extern "C" void pwreg_f32d7_optimize(void *regression, float *data)      { return pwreg_optimize<float, 7>(regression, data); }
// extern "C" void pwreg_f32d8_optimize(void *regression, float *data)      { return pwreg_optimize<float, 8>(regression, data); }
// extern "C" void pwreg_f32d9_optimize(void *regression, float *data)      { return pwreg_optimize<float, 9>(regression, data); }

extern "C" size_t pwreg_f64d1_optimize(void *regression, double *data)     { return pwreg_optimize<double, 1>(regression, data); }
extern "C" size_t pwreg_f64d2_optimize(void *regression, double *data)     { return pwreg_optimize<double, 2>(regression, data); }
extern "C" size_t pwreg_f64d3_optimize(void *regression, double *data)     { return pwreg_optimize<double, 3>(regression, data); }
extern "C" size_t pwreg_f64d4_optimize(void *regression, double *data)     { return pwreg_optimize<double, 4>(regression, data); }
extern "C" size_t pwreg_f64d5_optimize(void *regression, double *data)     { return pwreg_optimize<double, 5>(regression, data); }
extern "C" size_t pwreg_f64d6_optimize(void *regression, double *data)     { return pwreg_optimize<double, 6>(regression, data); }
extern "C" size_t pwreg_f64d7_optimize(void *regression, double *data)     { return pwreg_optimize<double, 7>(regression, data); }
extern "C" size_t pwreg_f64d8_optimize(void *regression, double *data)     { return pwreg_optimize<double, 8>(regression, data); }
extern "C" size_t pwreg_f64d9_optimize(void *regression, double *data)     { return pwreg_optimize<double, 9>(regression, data); }
extern "C" size_t pwreg_f64d16_optimize(void *regression, double *data)    { return pwreg_optimize<double, 16>(regression, data); }
extern "C" size_t pwreg_f64d32_optimize(void *regression, double *data)    { return pwreg_optimize<double, 32>(regression, data); }
extern "C" size_t pwreg_f64d64_optimize(void *regression, double *data)    { return pwreg_optimize<double, 64>(regression, data); }
extern "C" size_t pwreg_f64d128_optimize(void *regression, double *data)   { return pwreg_optimize<double, 128>(regression, data); }
extern "C" size_t pwreg_f64d256_optimize(void *regression, double *data)   { return pwreg_optimize<double, 256>(regression, data); }

#include <mvsr.h>
#include "mvsr.hpp"

template <typename Scalar>
inline void *mvsr_init(size_t samples, size_t dimensions, size_t variants, const Scalar *data, size_t minsegsize,
                       MvsrPlacement placement)
{
    if (placement != MvsrPlaceAll) return nullptr;

    auto *reg = new Mvsr<Scalar>(dimensions, variants);
    reg->placeSegments(data, samples, minsegsize);
    return reg;
}
template <typename Scalar>
inline size_t mvsr_reduce(void *reg, size_t minsegs, size_t maxsegs, MvsrAlg alg, MvsrMetric metric, MvsrScore score)
{
    if (metric != MvsrMetricMSE || score != MvsrScoreExact) return 0;

    auto *regression = reinterpret_cast<Mvsr<Scalar> *>(reg);
    switch (alg)
    {
    case MvsrAlgDP:
        regression->reduceDP(minsegs);
        break;
    case MvsrAlgGreedy:
        regression->reduceGreedy(minsegs);
        break;
    default:
        return 0;
    }

    return regression->getSegCount();
}
template <typename Scalar>
inline size_t mvsr_optimize(void *reg, const Scalar *data, unsigned int range, MvsrMetric metric)
{
    if (metric != MvsrMetricMSE) return 0;

    auto *regression = reinterpret_cast<Mvsr<Scalar> *>(reg);
    regression->optimize(data);
    return regression->getSegCount();
}
template <typename Scalar>
inline size_t mvsr_get_data(void *reg, size_t *breakpoints, Scalar *models, Scalar *errors)
{
    auto *regression = reinterpret_cast<Mvsr<Scalar> *>(reg);
    if (breakpoints != nullptr || models != nullptr || errors != nullptr)
    {
        size_t curbp = 0;
        size_t i = 0;
        for (auto &seg : regression->get())
        {
            if (breakpoints != nullptr) breakpoints[i] = curbp;
            if (models != nullptr)
                regression->getSegModel(seg, &models[i * (regression->getDimensions() * regression->getVariants())]);
            if (errors != nullptr) errors[i] = regression->getSegRss(seg);
            curbp += regression->getSegSize(seg);
            i++;
        }
    }
    return regression->getSegCount();
}
template <typename Scalar>
inline void *mvsr_copy(void *reg)
{
    auto *regression = reinterpret_cast<Mvsr<Scalar> *>(reg);
    auto *res = new Mvsr<Scalar>(*regression);
    return res;
}
template <typename Scalar>
inline void mvsr_release(void *reg)
{
    auto *regression = reinterpret_cast<Mvsr<Scalar> *>(reg);
    delete regression;
}

void *mvsr_init_f64(size_t samples, size_t dimensions, size_t variants, const double *data, size_t minsegsize,
                    MvsrPlacement placement)
{
    return mvsr_init<double>(samples, dimensions, variants, data, minsegsize, placement);
}

size_t mvsr_reduce_f64(void *reg, size_t minsegs, size_t maxsegs, MvsrAlg alg, MvsrMetric metric, MvsrScore score)
{
    return mvsr_reduce<double>(reg, minsegs, maxsegs, alg, metric, score);
}

size_t mvsr_optimize_f64(void *reg, const double *data, unsigned int range, MvsrMetric metric)
{
    return mvsr_optimize<double>(reg, data, range, metric);
}
size_t mvsr_get_data_f64(void *reg, size_t *breakpoints, double *models, double *errors)
{
    return mvsr_get_data<double>(reg, breakpoints, models, errors);
}
void *mvsr_copy_f64(void *reg)
{
    return mvsr_copy<double>(reg);
}
void mvsr_release_f64(void *reg)
{
    return mvsr_release<double>(reg);
}
void *mvsr_init_f32(size_t samples, size_t dimension, size_t variants, const float *data, size_t minsegsize,
                    MvsrPlacement placement)
{
    return mvsr_init<float>(samples, dimension, variants, data, minsegsize, placement);
}
size_t mvsr_reduce_f32(void *reg, size_t minsegs, size_t maxsegs, MvsrAlg alg, MvsrMetric metric, MvsrScore score)
{
    return mvsr_reduce<float>(reg, minsegs, maxsegs, alg, metric, score);
}
size_t mvsr_optimize_f32(void *reg, const float *data, unsigned int range, MvsrMetric metric)
{
    return mvsr_optimize<float>(reg, data, range, metric);
}
size_t mvsr_get_data_f32(void *reg, size_t *breakpoints, float *models, float *errors)
{
    return mvsr_get_data<float>(reg, breakpoints, models, errors);
}
void *mvsr_copy_f32(void *reg)
{
    return mvsr_copy<float>(reg);
}
void mvsr_release_f32(void *reg)
{
    return mvsr_release<float>(reg);
}

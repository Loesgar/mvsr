/** @file
 * Definition of the mvsr C-API
 */

#ifndef MVSR_H
#define MVSR_H

#include <stddef.h>

#ifdef __cplusplus
extern "C"
{
#endif

    /*------------------------------------------------------------------------------
                               Constants and Enumerations
    ------------------------------------------------------------------------------*/

    /** Strategy for initial placement. */
    enum MvsrPlacement
    {
        MvsrPlaceAll = 0 ///< Place as many segments as possible, given minsegsize
    };

    /** Algorithm used to reduce the segment count. */
    enum MvsrAlg
    {
        MvsrAlgGreedy = 0, ///< Fast greedy heuristic (minsegsize should be d)
        MvsrAlgDP = 1      ///< Slow dynamic program (optimal if minsegsize = 1)
    };

    /** Metric that is optimized by the algorithm. */
    enum MvsrMetric
    {
        MvsrMetricMSE = 0 ///< Minimize mean square error
    };

    /** Score to evaluate the segmentation and deduce the target segment count */
    enum MvsrScore
    {
        MvsrScoreExact = 0, ///< Always minimize exactly to the given segment number
        MvsrScoreCHI = 1    ///< Use adapted Calinski-Harabasz Index
    };

    /*------------------------------------------------------------------------------
                               Functions for doubles (f64)
    ------------------------------------------------------------------------------*/

    /**
     * @brief Initialize a regression object.
     *
     * @param samples    Number of samples
     * @param dimensions Number of input dimensions
     * @param variants   Number of output variants
     * @param data       Data matrix of all samples
     * @param minsegsize Minimum number of samples in any segment
     * @param placement  Placement strategy for initial segment placement
     *
     * @return The created regression object or NULL on error
     *
     * This function creates a regression object. The initial segments are later
     * reduced. Currently as many segments as possible are placed, ignoring the
     * placement parameter. This results in
     * $\lfloor\frac{samples}{minsegsize}\rfloor$ segments. The only segments that
     * may be larger than minsegsize is the last one. Parameter minsegsize should be
     * 'dimensions' if the greedy approach will be used and '1' if the dynamic
     * program will be used. The data contains all samples. Each sample is defined
     * by an array of $dimensions + variants$ values.
     */
    void *mvsr_init_f64(size_t samples, size_t dimensions, size_t variants, const double *data,
                        size_t minsegsize, MvsrPlacement placement);

    /**
     * @brief Reduce the amount of segments.
     *
     * @param reg     The already instantiated regression object.
     * @param minsegs Minimum number of target segments.
     * @param maxsegs Maximum number of segments (must be 0 if Exact is used).
     * @param alg     Algorithm to be used.
     * @param metric  Metric that should be optimized.
     * @param scoring Scoring to determine the segment count.
     *
     * @return The number of segments or '0' on error.
     *
     * This function reduces the number of segments. Depending on 'scoring', the
     * number is given by segments or dynamically determined.
     */
    size_t mvsr_reduce_f64(void *reg, size_t minsegs, size_t maxsegs, MvsrAlg alg,
                           MvsrMetric metric, MvsrScore score);

    /**
     * @brief Optimize the breakpoints.
     *
     * @param reg   The regression object.
     * @param data  The same data array used in the initialization.
     * @param range The size of the range in which the breakpoints are optimized.
     *
     * @return The number of segments or '0' on error.
     *
     * This should be executed after reducing with the greedy regression algorithm.
     * The range parameter is a number relative to the neighbouring segment size.
     */
    size_t mvsr_optimize_f64(void *reg, const double *data, unsigned int range, MvsrMetric metric);

    /**
     * @brief Get the regression results.
     *
     * @param      reg         The regression object.
     * @param[out] breakpoints Output array[segcount] for the breakpoint positions.
     * @param[out] models      Output array[segcount x variants] for the models.
     * @param[out] errors      Output array[segcount] for the segment errors.
     *
     * @return The number of segments.
     *
     * Get the data for all the segments. Each output parameter can be NULL to
     * ignore the parameter.
     */
    size_t mvsr_get_data_f64(void *reg, size_t *breakpoints, double *models, double *errors);

    /**
     * @brief Copy a regression.
     *
     * @param reg The regression to be copied.
     *
     * @return The new regression or NULL on error.
     *
     * Copies the regression with the exact same state. If this function does not
     * fail both regression objects can be used independently. Both regression
     * objects must be released individually.
     */
    void *mvsr_copy_f64(void *reg);

    /**
     * @brief Release a regression object.
     *
     * @param reg The regression object.
     *
     * This function releases a regression object and frees all associated
     * resources. The object MUST NOT be used afterwards by any function.
     */
    void mvsr_release_f64(void *reg);

    /*------------------------------------------------------------------------------
                               Functions for floats (f32)
    ------------------------------------------------------------------------------*/

    /**
     * @copydoc mvsr_init_f64
     */
    void *mvsr_init_f32(size_t samples, size_t dimension, size_t variants, const float *data,
                        size_t minsegsize, MvsrPlacement placement);

    /**
     * @copydoc mvsr_reduce_f64
     */
    size_t mvsr_reduce_f32(void *reg, size_t minsegs, size_t maxsegs, MvsrAlg alg,
                           MvsrMetric metric, MvsrScore score);

    /**
     * @copydoc mvsr_optimize_f64
     */
    size_t mvsr_optimize_f32(void *reg, const float *data, unsigned int range, MvsrMetric metric);

    /**
     * @copydoc mvsr_get_data_f64
     */
    size_t mvsr_get_data_f32(void *reg, size_t *breakpoints, float *models, float *errors);

    /**
     * @copydoc mvsr_copy_f64
     */
    void *mvsr_copy_f32(void *reg);

    /**
     * @copydoc mvsr_release_f64
     */
    void mvsr_release_f32(void *reg);

    /*------------------------------------------------------------------------------
                               Helper Functions
    ------------------------------------------------------------------------------*/

    inline size_t mvsr_f64(size_t samples, size_t dimensions, size_t variants, const double *data,
                           size_t minsegs, size_t maxsegs, MvsrAlg alg, MvsrScore score,
                           MvsrMetric metric, size_t *breakpoints, double *models, double *error)
    {
        size_t res = 0;
        size_t mss = (alg == MvsrAlgDP) ? 1 : dimensions;
        void *reg = mvsr_init_f64(samples, dimensions, variants, data, mss, MvsrPlaceAll);
        if (reg == NULL) return 0;
        // mvsr_optimize_f64(void *reg, const double *data, unsigned int range,
        //                   MvsrMetric metric) return 0;
        return 0;
    }

#ifdef __cplusplus
}
#endif

#endif // guard

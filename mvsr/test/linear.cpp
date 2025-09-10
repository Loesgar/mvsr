#include <mvsr.h>

double Data[] = 
{
    1.0,  1.0,  5.0,  1.0,
    1.0,  2.0,  5.0,  2.0,
    1.0,  3.0,  5.0,  3.0,
    1.0,  4.0,  5.0,  4.0,
    1.0,  5.0,  5.0,  5.0,
    1.0,  6.0,  5.0,  6.0,
    1.0,  7.0,  5.0,  7.0,
    1.0,  8.0,  5.0,  8.0,
    1.0,  9.0,  1.0,  8.0,
    1.0, 10.0,  2.0,  6.0,
    1.0, 11.0,  3.0,  4.0,
    1.0, 12.0,  4.0,  2.0,
    1.0, 13.0,  5.0,  0.0,
    1.0, 14.0,  6.0, -2.0,
    1.0, 15.0,  7.0, -4.0,
    1.0, 16.0,  9.0,  4.0,
    1.0, 17.0,  7.0,  4.0,
    1.0, 18.0,  5.0,  4.0,
    1.0, 19.0,  3.0,  4.0,
    1.0, 20.0,  1.0,  4.0,
};

int test_greedy()
{
    void *reg = mvsr_init_f64(20, 2, 2, Data, 2, MvsrPlaceAll);
    if (mvsr_reduce_f64(reg, 3, 3, MvsrAlgGreedy, MvsrMetricMSE, MvsrScoreExact) != 3)
    {
        mvsr_release_f64(reg);
        return 1;
    }
    if (mvsr_optimize_f64(reg, Data, -1, MvsrMetricMSE) != 3)
    {
        mvsr_release_f64(reg);
        return 2;
    }
    size_t bpOut[3];
    if (mvsr_get_data_f64(reg, bpOut, nullptr, nullptr) != 3)
    {
        mvsr_release_f64(reg);
        return 3;
    }
    if (bpOut[0] != 0 || bpOut[1] != 8 || bpOut[2] != 15)
    {
        mvsr_release_f64(reg);
        return 4;
    }
    mvsr_release_f64(reg);
    return 0;
}

int test_dp()
{
    void *reg = mvsr_init_f64(20, 2, 2, Data, 1, MvsrPlaceAll);
    if (mvsr_reduce_f64(reg, 3, 3, MvsrAlgDP, MvsrMetricMSE, MvsrScoreExact) != 3)
    {
        mvsr_release_f64(reg);
        return 1;
    }
    size_t bpOut[3];
    if (mvsr_get_data_f64(reg, bpOut, nullptr, nullptr) != 3)
    {
        mvsr_release_f64(reg);
        return 3;
    }
    if (bpOut[0] != 0 || bpOut[1] != 8 || bpOut[2] != 15)
    {
        mvsr_release_f64(reg);
        return 4;
    }
    mvsr_release_f64(reg);
    return 0;
}


int main()
{
    int res = test_greedy();
    if (res == 0)
    {
        res = test_dp();
        if (res != 0)
            res += 10;
    }
    return res;
}

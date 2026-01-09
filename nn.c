#define NN_IMPLEMENTATION
#include "nn.h"
#include <time.h>

float td_sum[] = {
    0, 0, 0, 0, 0, 0,
    0, 0, 0, 1, 0, 1,
    0, 0, 1, 0, 1, 0,
    0, 1, 0, 0, 0, 1,
    1, 0, 0, 0, 1, 0,
    0, 0, 1, 1, 1, 1,
    0, 1, 0, 1, 1, 0,
    1, 0, 0, 1, 1, 1,
    0, 1, 1, 0, 1, 1,
    1, 0, 1, 0, 0, 0,
    1, 1, 0, 0, 1, 1};

float td_xor[] = {
    0, 0, 0,
    0, 1, 1,
    1, 0, 1,
    1, 1, 0};

int main()
{
    srand(time(0));
    float *td = td_xor;
    size_t arch[] = {2, 2, 1};
    NN m = nn_alloc(arch, ARRAY_LEN(arch));
    NN g = nn_alloc(arch, ARRAY_LEN(arch));

    nn_rand(m, 0, 1.0f);

    size_t stride = 3;
    size_t n = 4;
    Mat X = {
        .rows = n,
        .cols = 2,
        .stride = stride,
        .es = td,
    };

    Mat Y = {
        .rows = n,
        .cols = 1,
        .stride = stride,
        .es = td + 2,
    };

    printf("cost before training: %.6f\n", nn_cost(m, X, Y));

    for (size_t epoch = 0; epoch < 1e5; epoch++)
    {
        nn_train(m, g, X, Y, 0.1f);
    }

    printf("cost after training: %.6f\n", nn_cost(m, X, Y));

    NN_PRINT(m);

    return 0;
}
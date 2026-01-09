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
    float *td = td_sum;
    size_t arch[] = {4, 4, 2};
    NN m = nn_alloc(arch, ARRAY_LEN(arch));
    NN g = nn_alloc(arch, ARRAY_LEN(arch));
    nn_rand(m, 0, 1.0f);

    size_t stride = 6;
    size_t n = 11;
    Mat X = {
        .rows = n,
        .cols = 4,
        .stride = stride,
        .es = td,
    };

    Mat Y = {
        .rows = n,
        .cols = 2,
        .stride = stride,
        .es = td + 4,
    };

    printf("cost before training: %.6f\n", nn_cost(m, X, Y));

    for (size_t epoch = 0; epoch < 1e5; epoch++)
    {
        nn_train(m, g, X, Y, 0.1f);
        if (epoch % 10000 == 0)
        {
            printf("Epoch %zu: cost = %.6f\n", epoch, nn_cost(m, X, Y));
        }
    }

    printf("cost after training: %.6f\n", nn_cost(m, X, Y));

    for (size_t i = 0; i < 11; i++)
    {
        for (size_t j = 0; j < 4; j++)
        {
            MAT_AT(NN_INPUT(m), 0, j) = MAT_AT(X, i, j);
        }
        nn_forward(m);
        printf("NN(");
        printf(") = (");
        for (size_t j = 0; j < 2; j++)
        {
            printf("%.4f", MAT_AT(NN_OUTPUT(m), 0, j));
            if (j < 1)
                printf(",");
        }
        printf(")\n");
    }
    return 0;
}
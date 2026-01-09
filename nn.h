#ifndef NN_H_
#define NN_H_

#include <stddef.h>
#include <stdio.h>
#include <math.h>

#ifndef NN_MALLOC
#include <stdlib.h>
#define NN_MALLOC calloc
#endif // NN_MALLOC

#ifndef NN_ASSERT
#include <assert.h>
#define NN_ASSERT assert
#endif // NN_ASSERT

#define ARRAY_LEN(x) (sizeof((x)) / sizeof((x)[0]))

typedef struct
{
    size_t rows;
    size_t cols;
    size_t stride;
    float *es;
} Mat;

typedef struct
{
    size_t count;
    Mat *ws;
    Mat *bs;
    Mat *as; // count + 1
} NN;

#define MAT_AT(m, i, j) ((m).es[(i) * (m).stride + (j)])
#define NN_INPUT(nn) ((nn).as[0])
#define NN_OUTPUT(nn) ((nn).as[(nn).count])

float sigmoid(float x);
void mat_sig(Mat m);

Mat mat_alloc(size_t rows, size_t cols);
void mat_rand(Mat m, float low, float high);
void mat_dot(Mat dst, Mat a, Mat b);
void mat_sum(Mat dst, Mat a);
void mat_print(Mat m, const char *name, size_t padding);
#define MAT_PRINT(m) mat_print(m, #m, 0)
void mat_fill(Mat m, float val);
Mat mat_row(Mat m, size_t row);
void mat_copy(Mat dst, Mat src);

NN nn_alloc(size_t *arch, size_t arch_count);
void nn_print(NN m, const char *name);
#define NN_PRINT(m) nn_print(m, #m)
void nn_rand(NN m, float low, float high);
void nn_forward(NN m);
float nn_cost(NN m, Mat X, Mat Y);
void nn_train(NN m, NN g, Mat X, Mat Y, float lr);

void nn_backprop(NN m, NN g, Mat X, Mat Y);

#endif // NN_H_

#ifdef NN_IMPLEMENTATION

// init mat = [0]
Mat mat_alloc(size_t rows, size_t cols)
{
    Mat m;
    m.rows = rows;
    m.cols = cols;
    m.stride = cols;
    m.es = NN_MALLOC(rows * cols, sizeof(*m.es));
    NN_ASSERT(m.es != NULL);
    return m;
}
void mat_dot(Mat dst, const Mat a, const Mat b)
{
    NN_ASSERT(a.cols == b.rows);
    size_t n = a.cols;
    NN_ASSERT(dst.rows == a.rows && dst.cols == b.cols);
    for (size_t i = 0; i < dst.rows; i++)
    {
        for (size_t j = 0; j < dst.cols; j++)
        {
            MAT_AT(dst, i, j) = 0.0f;
            for (size_t k = 0; k < n; k++)
            {
                MAT_AT(dst, i, j) += MAT_AT(a, i, k) * MAT_AT(b, k, j);
            }
        }
    }
}
void mat_sum(Mat dst, Mat a)
{
    NN_ASSERT(dst.rows == a.rows && dst.cols == a.cols);
    for (size_t i = 0; i < a.rows; i++)
    {
        for (size_t j = 0; j < a.cols; j++)
        {
            MAT_AT(dst, i, j) += MAT_AT(a, i, j);
        }
    }
}
void mat_print(Mat m, const char *name, size_t padding)
{
    printf("%*s%s = [\n", (int)padding, "", name);
    for (size_t i = 0; i < m.rows; i++)
    {
        printf("%*s", (int)padding, "");
        for (size_t j = 0; j < m.cols; j++)
        {
            float val = MAT_AT(m, i, j);
            printf("%f ", val);
        }
        printf("\n");
    }
    printf("%*s]\n", (int)padding, "");
}

void mat_rand(Mat m, float low, float high)
{
    for (size_t i = 0; i < m.rows; i++)
    {
        for (size_t j = 0; j < m.cols; j++)
        {
            MAT_AT(m, i, j) = (((float)rand() / (float)RAND_MAX) * (high - low)) + low;
        }
    }
}

void mat_fill(Mat m, float val)
{
    for (size_t i = 0; i < m.rows; i++)
    {
        for (size_t j = 0; j < m.cols; j++)
        {
            MAT_AT(m, i, j) = val;
        }
    }
}

float sigmoid(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

void mat_sig(Mat m)
{
    for (size_t i = 0; i < m.rows; i++)
    {
        for (size_t j = 0; j < m.cols; j++)
        {
            MAT_AT(m, i, j) = sigmoid(MAT_AT(m, i, j));
        }
    }
}

Mat mat_row(Mat m, size_t row)
{
    return (Mat){.rows = 1, .cols = m.cols, .stride = m.stride, .es = &MAT_AT(m, row, 0)};
}
void mat_copy(Mat dst, Mat src)
{
    NN_ASSERT(dst.rows == src.rows && dst.cols == src.cols);
    for (size_t i = 0; i < src.rows; i++)
    {
        for (size_t j = 0; j < src.cols; j++)
        {
            MAT_AT(dst, i, j) = MAT_AT(src, i, j);
        }
    }
}

NN nn_alloc(size_t *arch, size_t arch_count)
{
    NN m;
    m.count = arch_count - 1;
    m.ws = NN_MALLOC(m.count, sizeof(*m.ws));
    NN_ASSERT(m.ws != NULL);
    m.bs = NN_MALLOC(m.count, sizeof(*m.bs));
    NN_ASSERT(m.bs != NULL);
    m.as = NN_MALLOC(m.count + 1, sizeof(*m.as));
    NN_ASSERT(m.as != NULL);

    m.as[0] = mat_alloc(1, arch[0]); // input layer

    for (size_t i = 1; i < arch_count; i++)
    {
        m.ws[i - 1] = mat_alloc(m.as[i - 1].cols, arch[i]);
        m.bs[i - 1] = mat_alloc(1, arch[i]);
        m.as[i] = mat_alloc(1, arch[i]);
    }

    return m;
}
void nn_print(NN m, const char *name)
{
    char buf[256];
    printf("NN %s:\n[\n", name);
    for (size_t i = 0; i < m.count; i++)
    {
        snprintf(buf, sizeof(buf), "w%zu", i + 1);
        mat_print(m.ws[i], buf, 4);
        snprintf(buf, sizeof(buf), "b%zu", i + 1);
        mat_print(m.bs[i], buf, 4);
    }
    printf("]\n");
}

void nn_rand(NN m, float low, float high)
{
    for (size_t i = 0; i < m.count; i++)
    {
        mat_rand(m.ws[i], low, high);
        mat_rand(m.bs[i], low, high);
    }
}

void nn_forward(NN m)
{
    for (size_t i = 0; i < m.count; i++)
    {
        mat_dot(m.as[i + 1], m.as[i], m.ws[i]);
        mat_sum(m.as[i + 1], m.bs[i]);
        mat_sig(m.as[i + 1]);
    }
}

float nn_cost(NN m, Mat X, Mat Y)
{
    NN_ASSERT(X.rows == Y.rows);
    NN_ASSERT(Y.cols == NN_OUTPUT(m).cols);
    size_t n = X.rows;
    size_t q = Y.cols;

    float c = 0;

    for (size_t i = 0; i < n; i++)
    {
        Mat x = mat_row(X, i);
        Mat y = mat_row(Y, i);

        mat_copy(NN_INPUT(m), x);
        nn_forward(m);

        for (size_t j = 0; j < q; j++)
        {
            float diff = MAT_AT(NN_OUTPUT(m), 0, j) - MAT_AT(y, 0, j);
            c += diff * diff;
        }
    }

    return c / n;
}

void nn_backprop(NN m, NN g, Mat X, Mat Y)
{
    NN_ASSERT(X.rows == Y.rows);
    NN_ASSERT(NN_OUTPUT(m).cols == Y.cols);
    size_t n = X.rows;

    // i - current sample
    // l - current layer
    // j - current activation
    // k - previous activation

    for (size_t i = 0; i < n; i++)
    {
        mat_copy(NN_INPUT(m), mat_row(X, i));
        nn_forward(m);

        for (size_t j = 0; j <= m.count; j++)
        {
            mat_fill(g.as[j], 0);
        }

        for (size_t j = 0; j < Y.cols; j++)
        {
            MAT_AT(NN_OUTPUT(g), 0, j) = MAT_AT(NN_OUTPUT(m), 0, j) - MAT_AT(Y, i, j);
        }

        for (size_t l = m.count; l > 0; l--)
        {
            for (size_t j = 0; j < m.as[l].cols; j++)
            {
                float a = MAT_AT(m.as[l], 0, j);
                float da = MAT_AT(g.as[l], 0, j);
                float db = 2 * da * a * (1 - a);
                MAT_AT(g.bs[l - 1], 0, j) += db;
                for (size_t k = 0; k < m.as[l - 1].cols; k++)
                {
                    float pa = MAT_AT(m.as[l - 1], 0, k);
                    float pw = MAT_AT(m.ws[l - 1], k, j);
                    MAT_AT(g.ws[l - 1], k, j) += db * pa;
                    MAT_AT(g.as[l - 1], 0, k) += db * pw;
                }
            }
        }
    }

    // average gradients over all samples
    for (size_t l = 0; l < m.count; l++)
    {
        for (size_t i = 0; i < g.ws[l].rows; i++)
        {
            for (size_t j = 0; j < g.ws[l].cols; j++)
            {
                MAT_AT(g.ws[l], i, j) /= n;
            }
        }
        for (size_t i = 0; i < g.bs[l].rows; i++)
        {
            for (size_t j = 0; j < g.bs[l].cols; j++)
            {
                MAT_AT(g.bs[l], i, j) /= n;
            }
        }
    }
}

void nn_train(NN m, NN g, Mat X, Mat Y, float lr)
{
    nn_backprop(m, g, X, Y);

    for (size_t l = 0; l < m.count; l++)
    {
        for (size_t i = 0; i < m.ws[l].rows; i++)
        {
            for (size_t j = 0; j < m.ws[l].cols; j++)
            {
                MAT_AT(m.ws[l], i, j) -= lr * MAT_AT(g.ws[l], i, j);
            }
        }
        for (size_t i = 0; i < m.bs[l].rows; i++)
        {
            for (size_t j = 0; j < m.bs[l].cols; j++)
            {
                MAT_AT(m.bs[l], i, j) -= lr * MAT_AT(g.bs[l], i, j);
            }
        }
    }
}

#endif // NN_IMPLEMENTATION
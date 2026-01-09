#ifndef NN_H_
#define NN_H_

#include <stddef.h>
#include <stdio.h>
#include <math.h>

#ifndef NN_MALLOC
#include <stdlib.h>
#define NN_MALLOC malloc
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
float nn_finite_diff(NN m, NN g, float eps, Mat X, Mat Y);
void nn_train(NN m, NN g, Mat X, Mat Y, float lr);

void nn_zero(NN m);                         // 将梯度矩阵清零
void nn_backprop(NN m, NN g, Mat X, Mat Y); // 反向传播计算梯度

#endif // NN_H_

#ifdef NN_IMPLEMENTATION
Mat mat_alloc(size_t rows, size_t cols)
{
    Mat m;
    m.rows = rows;
    m.cols = cols;
    m.stride = cols;
    m.es = NN_MALLOC(sizeof(*m.es) * rows * cols);
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
    m.ws = NN_MALLOC(sizeof(*m.ws) * m.count);
    NN_ASSERT(m.ws != NULL);
    m.bs = NN_MALLOC(sizeof(*m.bs) * m.count);
    NN_ASSERT(m.bs != NULL);
    m.as = NN_MALLOC(sizeof(*m.as) * (m.count + 1));
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

float nn_finite_diff(NN m, NN g, float eps, Mat X, Mat Y)
{
    float saved;
    float c = nn_cost(m, X, Y);

    for (size_t l = 0; l < m.count; l++)
    {
        for (size_t i = 0; i < m.ws[l].rows; i++)
        {
            for (size_t j = 0; j < m.ws[l].cols; j++)
            {
                saved = MAT_AT(m.ws[l], i, j);
                MAT_AT(m.ws[l], i, j) = saved + eps;
                float c1 = nn_cost(m, X, Y);
                MAT_AT(g.ws[l], i, j) = (c1 - c) / (eps);
                MAT_AT(m.ws[l], i, j) = saved;
            }
        }
        for (size_t i = 0; i < m.bs[l].rows; i++)
        {
            for (size_t j = 0; j < m.bs[l].cols; j++)
            {
                saved = MAT_AT(m.bs[l], i, j);
                MAT_AT(m.bs[l], i, j) = saved + eps;
                float c1 = nn_cost(m, X, Y);
                MAT_AT(g.bs[l], i, j) = (c1 - c) / (eps);
                MAT_AT(m.bs[l], i, j) = saved;
            }
        }
    }

    return c;
}

void nn_train(NN m, NN g, Mat X, Mat Y, float lr)
{
    nn_finite_diff(m, g, 1e-4f, X, Y);

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

// 将神经网络的所有权重和偏置清零 (用于累积梯度前清空 g)
void nn_zero(NN m)
{
    for (size_t i = 0; i < m.count; i++)
    {
        mat_fill(m.ws[i], 0);
        mat_fill(m.bs[i], 0);
    }
}

// 反向传播算法实现
void nn_backprop(NN m, NN g, Mat X, Mat Y)
{
    NN_ASSERT(X.rows == Y.rows);
    size_t n = X.rows;

    // 1. 清空梯度网络 g
    nn_zero(g);

    // 遍历每一个样本 (SGD/Mini-batch)
    for (size_t i = 0; i < n; i++)
    {
        // 取出当前样本
        Mat    = mat_row(X, i);
        Mat y = mat_row(Y, i);

        // 2. 前向传播 (填充 m.as)
        mat_copy(NN_INPUT(m), x);
        nn_forward(m);

        // 3. 反向传播
        // 我们需要保存每一层的误差信号 (delta)
        // 注意：为了保持你的框架简洁，这里我们在循环内分配临时内存。
        // 在生产环境中，应该预分配这些内存以提高性能。

        // --- 输出层误差 ---
        // Cost Function: MSE = 0.5 * (a - y)^2
        // dC/da = (a - y)
        // Activation: Sigmoid
        // da/dz = a * (1 - a)
        // Delta L = (a - y) * a * (1 - a)

        Mat out = NN_OUTPUT(m);
        Mat delta = mat_alloc(out.rows, out.cols); // 1 x 10

        for (size_t j = 0; j < out.cols; j++)
        {
            float a = MAT_AT(out, 0, j);
            float target = MAT_AT(y, 0, j);
            // MSE derivative * Sigmoid derivative
            MAT_AT(delta, 0, j) = 2.0f * (a - target) * a * (1.0f - a);
        }

        // --- 反向遍历层 ---
        for (int l = m.count - 1; l >= 0; l--)
        {
            // 当前层的激活值 (上一层的输出)
            Mat current_act = m.as[l];

            // 计算权重梯度: dC/dw = delta * a_prev^T
            // 由于我们是行向量，这实际上是: a_prev^T * delta
            // g.ws[l] += current_act.T dot delta
            for (size_t r = 0; r < g.ws[l].rows; r++)
            {
                for (size_t c = 0; c < g.ws[l].cols; c++)
                {
                    float val = MAT_AT(current_act, 0, r) * MAT_AT(delta, 0, c);
                    MAT_AT(g.ws[l], r, c) += val;
                }
            }

            // 计算偏置梯度: dC/db = delta
            for (size_t c = 0; c < g.bs[l].cols; c++)
            {
                MAT_AT(g.bs[l], 0, c) += MAT_AT(delta, 0, c);
            }

            // 如果不是第一层，计算下一层（向前看是前一层）的误差 delta
            if (l > 0)
            {
                Mat prev_delta = mat_alloc(1, m.as[l].cols);
                Mat w = m.ws[l];

                // delta_prev = (delta dot w.T) * sigmoid_derivative
                for (size_t k = 0; k < w.rows; k++)
                {
                    float sum = 0.0f;
                    for (size_t j = 0; j < w.cols; j++)
                    {
                        sum += MAT_AT(delta, 0, j) * MAT_AT(w, k, j);
                    }
                    float a = MAT_AT(m.as[l], 0, k);
                    MAT_AT(prev_delta, 0, k) = sum * a * (1.0f - a);
                }

                free(delta.es);     // 释放旧 delta
                delta = prev_delta; // 更新 delta
            }
        }
        free(delta.es);
    }

    // 平均梯度
    for (size_t i = 0; i < g.count; i++)
    {
        for (size_t j = 0; j < g.ws[i].rows * g.ws[i].cols; j++)
            g.ws[i].es[j] /= n;
        for (size_t j = 0; j < g.bs[i].rows * g.bs[i].cols; j++)
            g.bs[i].es[j] /= n;
    }
}

#endif // NN_IMPLEMENTATION
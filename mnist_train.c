#define NN_IMPLEMENTATION
#include "nn.h"
#include <time.h>
#include <stdint.h>

// --- MNIST 加载工具 ---

uint32_t big_endian_to_little(uint32_t x)
{
    return ((x >> 24) & 0xff) | ((x << 8) & 0xff0000) |
           ((x >> 8) & 0xff00) | ((x << 24) & 0xff000000);
}

void load_mnist(const char *image_path, const char *label_path, Mat *images, Mat *labels, size_t count)
{
    FILE *f_img = fopen(image_path, "rb");
    FILE *f_lbl = fopen(label_path, "rb");

    if (!f_img || !f_lbl)
    {
        fprintf(stderr, "Failed to open dataset files.\n");
        exit(1);
    }

    uint32_t magic, num_imgs, rows, cols;
    fread(&magic, 4, 1, f_img);
    fread(&num_imgs, 4, 1, f_img);
    fread(&rows, 4, 1, f_img);
    fread(&cols, 4, 1, f_img);

    uint32_t magic_l, num_lbls;
    fread(&magic_l, 4, 1, f_lbl);
    fread(&num_lbls, 4, 1, f_lbl);

    size_t img_size = 28 * 28;

    // 分配内存
    *images = mat_alloc(count, img_size);
    *labels = mat_alloc(count, 10); // One-hot encoding

    uint8_t *temp_img = malloc(img_size);
    uint8_t temp_lbl;

    for (size_t i = 0; i < count; i++)
    {
        fread(temp_img, 1, img_size, f_img);
        fread(&temp_lbl, 1, 1, f_lbl);

        // 归一化图像 (0-255 -> 0.0-1.0)
        for (size_t j = 0; j < img_size; j++)
        {
            MAT_AT(*images, i, j) = temp_img[j] / 255.0f;
        }

        // One-hot label
        mat_fill(mat_row(*labels, i), 0.0f);
        MAT_AT(*labels, i, temp_lbl) = 1.0f;

        // 打印进度
        if ((i + 1) % 1000 == 0)
            printf("\rLoading... %zu/%zu", i + 1, count);
    }
    printf("\nData loaded.\n");

    free(temp_img);
    fclose(f_img);
    fclose(f_lbl);
}

// --- 预测准确率测试 ---
float accuracy(NN m, Mat X, Mat Y)
{
    size_t correct = 0;
    for (size_t i = 0; i < X.rows; i++)
    {
        Mat x = mat_row(X, i);
        mat_copy(NN_INPUT(m), x);
        nn_forward(m);

        // 找最大输出的索引
        size_t pred = 0;
        float max_val = -1;
        for (size_t j = 0; j < 10; j++)
        {
            if (MAT_AT(NN_OUTPUT(m), 0, j) > max_val)
            {
                max_val = MAT_AT(NN_OUTPUT(m), 0, j);
                pred = j;
            }
        }

        // 找真实标签索引
        size_t true_val = 0;
        for (size_t j = 0; j < 10; j++)
        {
            if (MAT_AT(Y, i, j) > 0.5f)
            { // 也就是 1.0
                true_val = j;
                break;
            }
        }

        if (pred == true_val)
            correct++;
    }
    return (float)correct / X.rows;
}

int main()
{
    srand(time(0));

    // 1. 设置网络架构
    // 输入: 784 (28x28), 隐藏层: 32, 输出: 10
    // 注意：隐藏层越大效果越好，但计算越慢。
    size_t arch[] = {784, 64, 10};
    NN m = nn_alloc(arch, ARRAY_LEN(arch));
    NN g = nn_alloc(arch, ARRAY_LEN(arch));

    nn_rand(m, -0.5f, 0.5f); // 稍微小一点的随机初始化范围

    // 2. 加载数据
    // 我们只加载部分数据进行演示 (例如 5000 张)，全部 60000 张训练会花一点时间
    size_t train_count = 5000;
    Mat train_X, train_Y;
    load_mnist("train-images.idx3-ubyte", "train-labels.idx1-ubyte", &train_X, &train_Y, train_count);

    // 3. 训练配置
    float lr = 0.5f;        // 学习率
    size_t batch_size = 32; // Mini-batch
    size_t epochs = 10;

    printf("Start training...\n");

    for (size_t epoch = 0; epoch < epochs; epoch++)
    {
        if (epoch > 5)
            lr = 0.1f;
        if (epoch > 8)
            lr = 0.01f;
        // 随机打乱数据通常更好，这里为了简单按顺序取 batch
        for (size_t i = 0; i < train_count; i += batch_size)
        {
            size_t size = (i + batch_size > train_count) ? (train_count - i) : batch_size;

            // 创建 Batch 的 View (引用内存，不拷贝)
            Mat batch_X = {
                .rows = size, .cols = train_X.cols, .stride = train_X.stride, .es = &MAT_AT(train_X, i, 0)};
            Mat batch_Y = {
                .rows = size, .cols = train_Y.cols, .stride = train_Y.stride, .es = &MAT_AT(train_Y, i, 0)};

            nn_train(m, g, batch_X, batch_Y, lr);
        }

        printf("Epoch %zu: Accuracy = %.2f%%\n", epoch + 1, accuracy(m, train_X, train_Y) * 100);
    }

    // 可以在这里加上测试集验证...

    // ---------------------------------------------------------
    // 4. 测试集验证 (Test Set Evaluation)
    // ---------------------------------------------------------
    printf("\n--------------------------------\n");
    printf("Loading Test Data...\n");

    // MNIST 标准测试集包含 10,000 张图片
    size_t test_count = 10000;
    Mat test_X, test_Y;

    // 加载测试文件 (文件名需与 shell 脚本下载的一致)
    load_mnist("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte", &test_X, &test_Y, test_count);

    printf("Evaluating on %zu test images...\n", test_count);

    // 使用同样的 accuracy 函数，但传入测试数据
    float test_acc = accuracy(m, test_X, test_Y);

    printf("--------------------------------\n");
    printf("FINAL TEST ACCURACY: %.2f%%\n", test_acc * 100.0f);
    printf("--------------------------------\n");

    // (可选) 单个样本推理演示：查看测试集第一张图的预测结果
    {
        printf("Demo: Inspecting the first image in test set:\n");
        Mat x = mat_row(test_X, 0);
        Mat y = mat_row(test_Y, 0);

        mat_copy(NN_INPUT(m), x);
        nn_forward(m);

        // 找出预测值
        size_t pred = 0;
        float max_val = -1;
        for (size_t j = 0; j < 10; j++)
        {
            float val = MAT_AT(NN_OUTPUT(m), 0, j);
            if (val > max_val)
            {
                max_val = val;
                pred = j;
            }
        }

        // 找出真实值
        size_t actual = 0;
        for (size_t j = 0; j < 10; j++)
        {
            if (MAT_AT(y, 0, j) > 0.5f)
            {
                actual = j;
                break;
            }
        }

        printf("  -> Model Prediction: %zu (Confidence: %.2f)\n", pred, max_val);
        printf("  -> Actual Label:     %zu\n", actual);

        if (pred == actual)
            printf("  -> Result: CORRECT\n");
        else
            printf("  -> Result: WRONG\n");
    }

    return 0;
}
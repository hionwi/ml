#include <stdio.h>

int main()
{
    printf("float td_mul[] = {\n");

    for (int a = 0; a < 16; a++)
    {
        for (int b = 0; b < 16; b++)
        {
            int product = a * b;

            // 1. 输入 A (4位)
            for (int i = 3; i >= 0; i--)
                printf("%d, ", (a >> i) & 1);

            // 2. 输入 B (4位)
            for (int i = 3; i >= 0; i--)
                printf("%d, ", (b >> i) & 1);

            // 3. 输出 Product (8位)
            for (int i = 7; i >= 0; i--)
            {
                printf("%d%s", (product >> i) & 1, (a == 15 && b == 15 && i == 0) ? "" : ", ");
            }

            printf("  // %d * %d = %d\n", a, b, product);
        }
    }
    printf("};\n");
    return 0;
}
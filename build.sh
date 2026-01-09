#!/bin/sh

set -xe

# clang -Wall -Wextra -o nn nn.c -lm -O3 && ./nn
clang -Wall -Wextra -o mnist_train mnist_train.c -O3 -ffast-math -march=native -lm && ./mnist_train
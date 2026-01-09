#!/bin/sh

set -xe

clang -Wall -Wextra -o gates gates.c -lm -O3 && ./gates
# clang -Wall -Wextra -o mnist_train mnist_train.c -lm -O3 && ./mnist_train
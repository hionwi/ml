#!/bin/sh

set -xe

# clang -Wall -Wextra -o nn nn.c -lm -O3 && ./nn
clang -Wall -Wextra -o nn nn.c -O3 -lm && ./nn
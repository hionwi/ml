#!/bin/bash

# 使用 Google 的存储镜像 (比原始官网更稳定)
BASE_URL="https://storage.googleapis.com/cvdf-datasets/mnist"

FILES=(
    "train-images-idx3-ubyte.gz"
    "train-labels-idx1-ubyte.gz"
    "t10k-images-idx3-ubyte.gz"
    "t10k-labels-idx1-ubyte.gz"
)

# 检查 wget 或 curl
if command -v wget >/dev/null 2>&1; then
    CMD="wget -c" # -c 支持断点续传
elif command -v curl >/dev/null 2>&1; then
    CMD="curl -L -O" # -L 跟随重定向
else
    echo "Error: Need wget or curl to download files."
    exit 1
fi

echo "Downloading MNIST dataset from Google Mirror..."

for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "$file already exists, skipping download."
    elif [ -f "${file%.gz}" ]; then
        # 检查是否已经解压并重命名过的文件存在 (比如 train-images.idx3-ubyte)
        # 这里做一个简单的检查，防止重复下载
        EXPECTED_NAME="${file//-/.}"
        EXPECTED_NAME="${EXPECTED_NAME%.gz}"
        if [ -f "$EXPECTED_NAME" ]; then
             echo "$EXPECTED_NAME found, skipping download."
        else
             echo "Downloading $file..."
             $CMD "$BASE_URL/$file"
        fi
    else
        echo "Downloading $file..."
        $CMD "$BASE_URL/$file"
    fi
done

echo "Extracting files..."
# 只有当有 .gz 文件时才解压
if ls *.gz 1> /dev/null 2>&1; then
    gzip -d *.gz
else
    echo "No .gz files to extract (maybe already extracted?)"
fi

echo "Renaming files to match C code expectations..."
# 映射: train-images-idx3-ubyte -> train-images.idx3-ubyte

declare -A name_map
name_map["train-images-idx3-ubyte"]="train-images.idx3-ubyte"
name_map["train-labels-idx1-ubyte"]="train-labels.idx1-ubyte"
name_map["t10k-images-idx3-ubyte"]="t10k-images.idx3-ubyte"
name_map["t10k-labels-idx1-ubyte"]="t10k-labels.idx1-ubyte"

for src in "${!name_map[@]}"; do
    dest="${name_map[$src]}"
    if [ -f "$src" ]; then
        mv "$src" "$dest"
        echo "Renamed $src -> $dest"
    elif [ -f "$dest" ]; then
        echo "$dest is ready."
    else
        echo "Warning: $src not found. (Download might have failed)"
    fi
done

echo "Done! You are ready to compile."
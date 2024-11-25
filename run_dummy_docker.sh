#!/bin/bash

# CUDA 버전 확인 (nvidia-smi 사용)
cuda_version=$(nvidia-smi | grep -oP "CUDA Version: \K[0-9]+")

# 도커 이미지 선택
if [ "$cuda_version" -eq 12 ]; then
    docker_image="bic4907/pcgrl:cu12"
elif [ "$cuda_version" -eq 11 ]; then
    docker_image="bic4907/pcgrl:cu11"
else
    echo "Unsupported CUDA version: $cuda_version"
    exit 1
fi

# 도커 컨테이너 실행
docker run --rm -it --gpus all -v $(pwd):/app -v /mnt/nas:/mnt/nas --network host $docker_image /bin/bash

#!/bin/bash

# Docker 기반 실행 스크립트 (빈 GPU 자동 할당 및 로그 저장)
# 사용법: ./run_in_docker_with_gpu.sh python_file key1=value1 key2=value2 ...

# 첫 번째 인자로 Python 스크립트 파일 이름을 받음
python_file="$1"
shift  # 첫 번째 인자 제거

# 나머지 인자들을 key=value 형태로 처리
ARGS=("$@")

# key=value 형식의 인자를 공백으로 구분하여 Python 스크립트 실행에 전달
param_str=""
for arg in "${ARGS[@]}"; do
    param_str+="$arg "
done

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

# 로그 디렉토리 설정
mkdir -p output_logs error_logs
timestamp=$(date +"%Y%m%d_%H%M%S")
log_file="output_logs/output_${timestamp}.log"
error_log_file="error_logs/error_${timestamp}.log"

# GPU 선택 로직
echo "Searching for available GPU..."

# `nvidia-smi`로 GPU 메모리 사용량 확인
gpu_info=$(nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader,nounits)
available_gpu=$(echo "$gpu_info" | awk -F, '{if ($5 > 0) print $1 " " $5}' | sort -k2 -nr | head -n1 | cut -d' ' -f1)

if [ -z "$available_gpu" ]; then
    echo "No available GPU found!" | tee -a "$error_log_file"
    exit 1
fi

# 선택된 GPU 상세 정보 가져오기
selected_gpu_info=$(echo "$gpu_info" | awk -v gpu_id="$available_gpu" -F, '{if ($1 == gpu_id) print}')
selected_gpu_name=$(echo "$selected_gpu_info" | cut -d, -f2)
selected_gpu_total_mem=$(echo "$selected_gpu_info" | cut -d, -f3)
selected_gpu_used_mem=$(echo "$selected_gpu_info" | cut -d, -f4)
selected_gpu_free_mem=$(echo "$selected_gpu_info" | cut -d, -f5)

# 선택된 GPU 출력
echo "Selected GPU: $available_gpu (GPU Number: $available_gpu)" | tee -a "$log_file"
echo "GPU Details:" | tee -a "$log_file"
echo "  GPU ID: $available_gpu" | tee -a "$log_file"
echo "  Model Name: $selected_gpu_name" | tee -a "$log_file"
echo "  Total Memory: ${selected_gpu_total_mem}MiB" | tee -a "$log_file"
echo "  Used Memory: ${selected_gpu_used_mem}MiB" | tee -a "$log_file"
echo "  Free Memory: ${selected_gpu_free_mem}MiB" | tee -a "$log_file"


# 제외할 인자 목록
exclude_args=("wandb_project" "n_envs" "overwrite")

# 컨테이너 이름 초기화
container_name="pcgrllm"

for arg in "$@"; do
    # 인자의 이름과 값을 '='로 분리
    key=$(echo "$arg" | cut -d '=' -f 1)

    # 제외할 인자가 아닌 경우에만 처리
    exclude=false
    for exclude_arg in "${exclude_args[@]}"; do
        if [[ "$key" == "$exclude_arg" ]]; then
            exclude=true
            break
        fi
    done

    if [[ "$exclude" == false ]]; then
        # '='를 '-'로, '.'를 '_'로 대체
        sanitized_arg=$(echo "$arg" | sed 's/=/-/g; s/\./_/g')
        container_name="${container_name}_${sanitized_arg}"
    fi
done

echo "Container Name: $container_name"
echo "Output Log File: $log_file"

# Docker 실행 명령어
docker_command="docker run --rm -it
    -v $(pwd):/workspace
    -w /workspace
    --gpus all
    -e CUDA_VISIBLE_DEVICES=$available_gpu
    -v $(pwd)/.netrc:/.netrc
    --network=host
    -e HF_HOME=/workspace/cache/huggingface
    --name \"$container_name\"
    -u $(id -u):$(id -g)
    $docker_image
    python $python_file $param_str"

echo "Executing Docker command:" | tee -a "$log_file"
echo "$docker_command" | tee -a "$log_file"

# Docker 명령 실행 및 로그 기록
{
    eval $docker_command
} 2>&1 | tee -a "$log_file"

# 실행 결과 확인
exit_code=$?
if [ $exit_code -ne 0 ]; then
    echo "Execution failed. Check logs for details." | tee -a "$error_log_file"
    echo "Docker logs (last 10 lines of $log_file):" | tee -a "$error_log_file"
    tail -n 10 "$log_file" | tee -a "$error_log_file"
    exit $exit_code
else
    echo "Execution completed successfully." | tee -a "$log_file"
fi

exit $exit_code

#!/bin/bash

#SBATCH --job-name=multi_pcgrl  # 작업 이름
#SBATCH --output=output_logs/output_%A_%a.out    # 출력 파일 경로 (%A: Job ID, %a: array task ID)
#SBATCH --error=error_logs/error_%A_%a.err       # 에러 파일 경로
#SBATCH --ntasks=1                        # 실행할 작업 수
#SBATCH --cpus-per-task=4                 # 각 작업당 사용할 CPU 코어 수
#SBATCH --gres=gpu:1                      # GPU 자원 설정 (1개의 GPU 사용)
#SBATCH --mem=16G                         # 각 작업당 메모리 할당
#SBATCH --time=1-00:00:00                 # 최대 실행 시간 (1일)
#SBATCH --account=pr_174_general          # SLURM 계정 설정


# 첫 번째 인자로 파이썬 파일 이름을 받음
python_file="$1"

# 나머지 인자들 (key=value 형태로 전달된 여러 조건들)
ARGS=("${@:2}")  # 첫 번째 인자 제외


# 전달된 인자를 파싱하여 key=value 형식으로 처리
keys=()            # key 배열 (예: seed, gpt_model 등)
values_array=()    # 각 key에 대한 value 배열 리스트

# 전달된 인자들을 파싱해서 key-value 배열로 나눔
for arg in "${ARGS[@]}"; do
    key=$(echo $arg | cut -d'=' -f1)
    value=$(echo $arg | cut -d'=' -f2)

    # key가 새로운 값이면 추가
    if [[ ! " ${keys[*]} " =~ " ${key} " ]]; then
        keys+=("$key")
        values_array+=("$value")
    else
        # 이미 존재하는 key면 값을 추가 (콤마로 구분된 경우)
        idx=$(echo "${keys[@]}" | tr ' ' '\n' | grep -n -w "$key" | cut -d: -f1)
        idx=$((idx-1))
        values_array[$idx]+=",$value"
    fi
done

# 각 key에 대한 value 조합을 생성 (가능한 모든 조합 만들기)
PARAMS_ARRAY=()

# 콤마로 구분된 각 key에 대한 value 조합 생성
generate_combinations() {
    local array=("$@")
    local num_keys=${#keys[@]}

    # 초기에는 첫 번째 key의 값을 넣음
    combinations=()
    IFS=',' read -r -a values <<< "${array[0]}"
    for value in "${values[@]}"; do
        combinations+=("$value")
    done

    # 각 키의 값들을 조합하여 combinations 배열에 추가
    for ((i=1; i<num_keys; i++)); do
        local new_combinations=()
        IFS=',' read -r -a values <<< "${array[i]}"
        for combination in "${combinations[@]}"; do
            for value in "${values[@]}"; do
                new_combinations+=("$combination,$value")
            done
        done
        combinations=("${new_combinations[@]}")
    done

    # 최종 조합 반환
    echo "${combinations[@]}"
}

# 동적으로 생성한 모든 key-value 조합 생성
param_combinations=($(generate_combinations "${values_array[@]}"))

# 각 조합에 대해 key=value 형식으로 PARAMS_ARRAY 생성
for combination in "${param_combinations[@]}"; do
    IFS=',' read -r -a values <<< "$combination"
    param_str=""
    for ((i=0; i<${#keys[@]}; i++)); do
        param_str+="${keys[$i]}=${values[$i]},"
    done
    param_str=$(echo "$param_str" | sed 's/,$//')  # 마지막 콤마 제거
    PARAMS_ARRAY+=("$param_str")
done

# SLURM 작업 배열 크기 설정
num_experiments=${#PARAMS_ARRAY[@]}
array_range="1-${num_experiments}"

# SLURM_ARRAY_TASK_ID로 실행되는지 확인
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    # SLURM 작업이 아닌 경우 -> sbatch가 없을 때 에러를 발생시키지 않고, 명령만 출력
    if command -v sbatch &> /dev/null; then
        # SLURM 작업 제출 (현재 스크립트를 재실행하며 --array 옵션을 동적으로 설정)
        echo "Submitting jobs with array range: $array_range"
        echo sbatch --array=$array_range "$0" "$1" "${ARGS[@]}"
        sbatch --array=$array_range "$0" "$1" "${ARGS[@]}"
    else
        echo "sbatch not found, displaying what would be executed."
        for i in "${!PARAMS_ARRAY[@]}"; do
            echo "Experiment $((i+1)): ${PARAMS_ARRAY[$i]}"
        done
    fi
    exit 0
fi

# SLURM 작업으로 실행되는 경우 (SLURM_ARRAY_TASK_ID 기반)
PARAMS=${PARAMS_ARRAY[$SLURM_ARRAY_TASK_ID-1]}

# exp_name 생성 (key-value 형식으로 조합하여 _로 구분)
IFS=',' read -r -a param_kv <<< "$PARAMS"
exp_name=""
for kv in "${param_kv[@]}"; do
  key=$(echo $kv | cut -d'=' -f1)
  value=$(echo $kv | cut -d'=' -f2)
  if [[ "$key" != "n_envs" && "$key" != "overwrite" && "$key" != "seed" ]]; then  # n_envs와 overwrite는 exp_name에 포함되지 않음
    exp_name+="${key}-${value}_"
  fi
done
# 마지막 _ 제거
exp_name=$(echo $exp_name | sed 's/_$//')

echo "Running experiment with parameters: $PARAMS"
if [ -n "$exp_name" ]; then
    exp_name_arg="exp_name=$exp_name"
else
    exp_name_arg=""
fi

# Python 스크립트 실행
python_command="python $python_file ${PARAMS//,/ } $exp_name_arg +wandb_key=ecd68e29f83835871f85d88e6f1b9e16bb93ff1c"
echo "Python command to be executed: $python_command"
$python_command
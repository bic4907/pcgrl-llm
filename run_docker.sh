#!/bin/bash

# 외부에서 받은 인자들 (공백으로 구분된 key=value 쌍)
INPUT_ARGS="$@"

# 콤마로 구분된 key=value 쌍을 파싱
args_array=()
for kv_pair in "$@"; do
    args_array+=("$kv_pair")
done

# exp_name 생성 (모든 key=val 형식, 각 인자값 사이에 _로 구분)
exp_name=""
for arg in "${args_array[@]}"; do
    exp_name+="${arg}_"
done

# 마지막 _ 제거 (substring expression 수정)
exp_name=$(echo $exp_name | sed 's/_$//')

echo "Running experiment with exp_name=$exp_name on GPU"

# Python 명령 생성 (exp_name이 존재할 경우에만 추가)
if [[ -n "$exp_name" ]]; then
    python_command="python experiment.py ${INPUT_ARGS//,/ } exp_name=$exp_name"
else
    python_command="python experiment.py ${INPUT_ARGS//,/ }"
fi

# Python 실행 명령을 출력
echo "Python command to be executed: $python_command"

# sbatch가 있는지 확인하고 없으면 그냥 명령을 출력하고 종료
if command -v sbatch &> /dev/null; then
    # sbatch 옵션을 동적으로 추가하여 job-name 설정
    sbatch --job-name=$exp_name --output=output_logs/output_%A_%a.out --error=error_logs/error_%A_%a.err <<-EOF
#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --array=1-3
#SBATCH --account=pr_174_general

# 환경변수 설정
export LOG_LEVEL=DEBUG
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Python 스크립트 실행 (파싱된 모든 인자 전달)
echo "Executing: $python_command"
$python_command
EOF
else
    echo "sbatch not found, showing command instead of execution:"
    echo "#SBATCH --job-name=$exp_name --output=output_logs/output_%A_%a.out --error=error_logs/error_%A_%a.err"
    echo "Executing the following command instead: $python_command"
fi

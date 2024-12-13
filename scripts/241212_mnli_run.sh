filename=$(basename "$0")
filename="${filename%.*}"

python3 main.py train mnli_debug_config 2>&1 | tee results/${filename}.log


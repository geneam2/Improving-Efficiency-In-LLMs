filename=$(basename "$0")
filename="${filename%.*}"

python3 main.py train cola_roberta_config 2>&1 | tee results/${filename}.log


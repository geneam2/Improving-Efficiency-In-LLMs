# python3 main.py train mnli_debug_config_2 2>&1 | tee scripts/mnli_debug_config_2.log
python3 main.py train cola_debug_config_2 2>&1 | tee scripts/cola_debug_config_2.log
python3 main.py train mrpc_debug_config_2 2>&1 | tee scripts/mrpc_debug_config_2.log
python3 main.py train QNLI_debug_config_2 2>&1 | tee scripts/QNLI_debug_config_2.log
python3 main.py train qqp_debug_config_2 2>&1 | tee scripts/qqp_debug_config_2.log

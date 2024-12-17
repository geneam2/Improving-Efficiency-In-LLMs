python3 main.py train configs_baseline.cola_roberta_config 2>&1 | tee configs_baseline/cola_roberta_config.log
python3 main.py train configs_baseline.mnli_roberta_config 2>&1 | tee configs_baseline/mnli_roberta_config.log
python3 main.py train configs_baseline.mrpc_roberta_config 2>&1 | tee configs_baseline/mrpc_roberta_config.log
python3 main.py train configs_baseline.qnli_roberta_config 2>&1 | tee configs_baseline/qnli_roberta_config.log
python3 main.py train configs_baseline.qqp_roberta_config 2>&1 | tee configs_baseline/qqp_roberta_config.log

python3 main.py train configs_lora.cola_roberta_config_lora 2>&1 | tee configs_lora/cola_roberta_config_lora.log
python3 main.py train configs_lora.mnli_roberta_config_lora 2>&1 | tee configs_lora/mnli_roberta_config_lora.log
python3 main.py train configs_lora.mrpc_roberta_config_lora 2>&1 | tee configs_lora/mrpc_roberta_config_lora.log
python3 main.py train configs_lora.qnli_roberta_config_lora 2>&1 | tee configs_lora/qnli_roberta_config_lora.log
python3 main.py train configs_lora.qqp_roberta_config_lora 2>&1 | tee configs_lora/qqp_roberta_config_lora.log
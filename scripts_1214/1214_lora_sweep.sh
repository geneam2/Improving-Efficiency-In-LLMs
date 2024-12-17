python3 sweep_train.py train configs_sweep_lora.cola_roberta_config_lora 2>&1 | tee configs_sweep_lora/cola_roberta_config_lora.log
python3 sweep_train.py train configs_sweep_lora.qqp_roberta_config_lora 2>&1 | tee configs_sweep_lora/qqp_roberta_config_lora.log
python3 sweep_train.py train configs_sweep_lora.qnli_roberta_config_lora 2>&1 | tee configs_sweep_lora/qnli_roberta_config_lora.log
python3 sweep_train.py train configs_sweep_lora.mrpc_roberta_config_lora 2>&1 | tee configs_sweep_lora/mrpc_roberta_config_lora.log
python3 sweep_train.py train configs_sweep_lora.mnli_roberta_config_lora 2>&1 | tee configs_sweep_lora/mnli_roberta_config_lora.log
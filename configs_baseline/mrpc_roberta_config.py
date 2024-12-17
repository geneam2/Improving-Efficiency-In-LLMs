import os
from datetime import datetime
# roberta
# epochs: 10 w/ early stopping
# lr ∈ {1e−5, 2e−5, 3e−5}
# bsz ∈ {16, 32}
# warmup ratio : 0.06

class task:
    model = "SequenceClassificationModel"
    model_name = "FacebookAI/roberta-base"
    task_name = "MRPC"

class train:
    learning_rate = 1e-5
    epochs = 5
    weight_decay = 0.01
    report_to = "wandb"
    val_batch = 32
    test_batch = 32
    train_batch = 32
    warmup_ratio = 0.06
    grad_accum = 1
    scheduler = "InverseSqrt"
    max_seq_len = 512    

class wandb_config:
    project_name = "jjh"
    experiment_name = f"roberta_{task.task_name}_debug_invsqrt_{datetime.now().strftime('%H_%M_%S_%m%d')}"
    api_key_path = f"{os.path.dirname(os.path.realpath(__file__))}/wandb_api.local"
    api_key = open(api_key_path).readline()

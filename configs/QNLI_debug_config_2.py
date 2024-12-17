import os
from datetime import datetime

class task:
    model = "SequenceClassificationModel"
    model_name = "distilbert-base-uncased"
    task_name = "QNLI"

class train:
    learning_rate = 2e-04
    epochs = 5
    weight_decay = 0.01
    report_to = "wandb"
    val_batch = 64
    test_batch = 64
    train_batch = 128
    warmup_ratio = 0.1
    grad_accum = 1
    scheduler = "InverseSqrt"

class wandb_config:
    project_name = "jjh"
    experiment_name = f"{task.task_name}_debug_invsqrt_{datetime.now().strftime('%H_%M_%S_%m%d')}"
    api_key_path = f"{os.path.dirname(os.path.realpath(__file__))}/wandb_api.local"
    api_key = open(api_key_path).readline()

# class train:
#     output_dir = "debug"
#     logging_strategy = "steps"
#     eval_strategy = "epoch"
#     save_strategy = "epoch"
#     learning_rate = 3e-5
#     num_train_epochs = 3
#     per_device_train_batch_size = 16
#     per_device_eval_batch_size = 16
#     weight_decay = 0.01
#     logging_dir = "logs"
#     logging_steps = 10
#     report_to = "wandb"
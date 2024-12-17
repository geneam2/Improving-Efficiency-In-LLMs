# all three train, eval, data classes MUST be implemented even if any one of them is empty
import os
from datetime import datetime

class task:
    model = "QuestionAnsweringModel"
    model_name = "distilbert-base-uncased"
    task_name = "SQuADv2"

class train:
    learning_rate = 1e-5
    epochs = 3 # 5
    weight_decay = 0.01
    report_to = "wandb"
    val_batch = 32
    test_batch = 32
    train_batch = 64 # 8
    warmup_ratio = 0.06
    grad_accum = 1
    scheduler = "InverseSqrt"
    max_seq_len = 512 
    resume_from_checkpoint = False # Eventually, we would want to do checkpointing

class wandb_config:
    project_name = "squad_distilbert"
    experiment_name = f"{project_name}_{task.task_name}_{datetime.now().strftime('%H_%M_%S_%m%d')}"
    api_key_path = f"{os.path.dirname(os.path.realpath(__file__))}/wandb_api.local"
    api_key = open(api_key_path).readline()

class eval:
    pass

class data:
    pass

from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType

# all three train, eval, data classes MUST be implemented even if any one of them is empty


class model:

    model_name = "distilbert-base-uncased"
    task = "SQuADV2"


class train:
    output_dir = "11/20 distilbert"
    logging_strategy = "steps"
    eval_strategy = "epoch"
    save_strategy = "epoch"
    learning_rate = 3e-5
    num_train_epochs = 3
    per_device_train_batch_size = 16
    per_device_eval_batch_size = 16
    weight_decay = 0.01
    logging_dir = "logs"
    logging_steps = 10
    report_to = "wandb"


class eval:
    pass


class data:
    pass

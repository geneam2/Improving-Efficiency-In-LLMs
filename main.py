import sys
import importlib

from custom_trainer import CustomTrainer
from utils import (
    MODEL_REGISTRY,
    TASK_REGISTRY,
    TRAINER_REGISTRY,
    read_config,
)

importlib.import_module("model")
importlib.import_module("task")

def main_train(config):
    config_path = f"configs.{config}" if "." not in config else config
    args = read_config(config_path)
    task_class = TASK_REGISTRY.get(args['task'].task_name)
    model_fn = MODEL_REGISTRY.get(args['task'].model)
    task = task_class(args['task'], args['train'], model_fn)
    trainer = CustomTrainer(task, args.get("wandb_config", None))
    trainer.train(args['train'])

def main_eval(config):
    pass

def main_infer(config):
    pass

if __name__=="__main__":
    assert len(sys.argv) == 3, "define mode (train | eval) and config"
    print("Executing python3", sys.argv)
    mode = sys.argv[1]
    config = sys.argv[2]
    if mode == "train":
        main_train(config)
    elif mode == "eval":
        main_eval(config)
    elif mode == "infer":
        main_infer(config)

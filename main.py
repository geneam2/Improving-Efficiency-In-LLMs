import sys

from utils import (
    MODEL_REGISTRY,
    TASK_REGISTRY,
    TRAINER_REGISTRY,
    read_config,
)

def main(mode):
    model_config, train_config, eval_config, data_config = read_config(f"configs.{mode}")
    model_class = getattr(MODEL_REGISTRY, mode)
    task_class = getattr(TASK_REGISTRY, model_config.task)
    model = model_class(model_config)
    task = task_class(model)
    task.train(train_config)
    task.evaluate(eval_config)

if __name__=="__main__":
    assert len(sys.argv) == 2, "You must input the mode"
    print("Executing python3", sys.argv)
    main(sys.argv[2])


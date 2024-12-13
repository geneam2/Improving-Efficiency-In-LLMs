import json
import inspect
import importlib

MODEL_REGISTRY = {}
TASK_REGISTRY = {}
TRAINER_REGISTRY = {}

def register_classes(class_obj, registry: dict):
    assert class_obj.__name__ not in registry, "{} has duplicate class object names, this is not permitted!".format(class_obj.__name__)
    registry[class_obj.__name__] = class_obj

    return registry

def register_to(registry):
    def register_to_inner(class_obj):
        nonlocal registry
        register_classes(class_obj, registry)
    return register_to_inner


def read_config(path):
    class Args():
        built_in = "__"
        def __init__(self, config):
            for k, i in config.__dict__.items():
                if k[:2] == k[-2:] == self.built_in:
                    # clear built-in modules
                    continue
                setattr(self, k, i)

    config = importlib.import_module(path)

    args = dict()
    for name, obj in inspect.getmembers(config):
        if inspect.isclass(obj) and obj.__module__ == config.__name__:
            args[name] = Args(obj)

    # return Args(config.model), Args(config.train), Args(config.task)
    return args

if __name__=="__main__":
    # example
    model_config, train_config, eval_config, data_config = read_config("configs.config")

    print("train_config", train_config.__dict__)
    print("eval_config", eval_config.__dict__)
    print("data_config", data_config.__dict__)

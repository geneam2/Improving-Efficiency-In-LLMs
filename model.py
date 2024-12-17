from peft import LoraConfig, TaskType, get_peft_model
from datasets import load_dataset
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

from utils import register_to, MODEL_REGISTRY

@register_to(MODEL_REGISTRY)
def SequenceClassificationModel(model_name, **kwargs):
    return AutoModelForSequenceClassification.from_pretrained(model_name, **kwargs)


@register_to(MODEL_REGISTRY)
def SequenceClassificationLoRA(model_name, lora_r, lora_alpha, **kwargs):
    model = AutoModelForSequenceClassification.from_pretrained(model_name, **kwargs)
    lora_config = LoraConfig(
        r=lora_r,
        target_modules=["query", "value"],
        task_type=TaskType.SEQ_CLS,
        lora_alpha=lora_alpha,
        lora_dropout=0.1,
    )
    return get_peft_model(model, lora_config)

@register_to(MODEL_REGISTRY)
def QuestionAnsweringModel(model_name, **kwargs):
    return AutoModelForQuestionAnswering.from_pretrained(model_name)

@register_to(MODEL_REGISTRY)
def QuestionAnsweringModelLoRA(model_name, **kwargs):
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    lora_config = LoraConfig(
        r=8,
        target_modules=["query", "value"],
        task_type=TaskType.QUESTION_ANS,
        lora_alpha=8,
        lora_dropout=0.1,
    )

    return get_peft_model(model, lora_config)

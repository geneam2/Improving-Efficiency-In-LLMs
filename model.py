
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
def SequenceClassificationLoRA(model_name, **kwargs):
    return AutoModelForSequenceClassification.from_pretrained(model_name, **kwargs)

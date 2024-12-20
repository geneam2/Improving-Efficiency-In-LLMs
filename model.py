from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig
)
import torch

from utils import register_to, MODEL_REGISTRY


def print_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('Model size: {:.3f}MB'.format(size_all_mb))


@register_to(MODEL_REGISTRY)
def SequenceClassificationModel(model_name, **kwargs):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, **kwargs)

    print_model_size(model)
    return model


@register_to(MODEL_REGISTRY)
def SequenceClassificationLoRA(model_name, lora_r, lora_alpha, **kwargs):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, **kwargs)
    lora_config = LoraConfig(
        r=lora_r,
        target_modules=["query", "value"],
        task_type=TaskType.SEQ_CLS,
        lora_alpha=lora_alpha,
        lora_dropout=0.1,
    )

    print_model_size(model)
    return get_peft_model(model, lora_config)


@register_to(MODEL_REGISTRY)
def QuestionAnsweringModel(model_name, **kwargs):
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)

    print_model_size(model)
    return model


@register_to(MODEL_REGISTRY)
def QuestionAnsweringModelLoRA(model_name, lora_r, lora_alpha, **kwargs):
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    lora_config = LoraConfig(
        r=lora_r,
        target_modules=["query", "value"],
        task_type=TaskType.QUESTION_ANS,
        lora_alpha=lora_alpha,
        lora_dropout=0.1,
    )

    print_model_size(model)

    return get_peft_model(model, lora_config)


@register_to(MODEL_REGISTRY)
def QuestionAnsweringModelQLoRA(model_name, lora_r, lora_alpha, **kwargs):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    start_model = AutoModelForQuestionAnswering.from_pretrained(
        model_name, quantization_config=bnb_config,
        device_map={"": 0}
    )

    print_model_size(start_model)

    lora_config = LoraConfig(
        task_type=TaskType.QUESTION_ANS,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=['attention.q_lin',
                        'attention.k_lin', 'attention.v_lin']
    )
    # prepare int-8 model for training
    start_model.gradient_checkpointing_enable()
    start_model = prepare_model_for_kbit_training(start_model)

    model = get_peft_model(start_model, lora_config)
    model.print_trainable_parameters()
    for name, param in model.named_parameters():
        if "qa_outputs" in name:
            param.requires_grad = True

    return model

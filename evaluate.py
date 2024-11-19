
import os
import sys

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, Trainer
from utils import device
import torch

# Load the dataset, model, and tokenizer
dataset = load_dataset("squad_v2")
train_dataset = dataset["train"]
valid_dataset = dataset["validation"]

saved_weights_folder = os.path.join(
    os.path.dirname(__file__), "saved_weights")

print("Weights folder:", saved_weights_folder)
model = AutoModelForQuestionAnswering.from_pretrained(saved_weights_folder)
model.to(device)
tokenizer = AutoTokenizer.from_pretrained(saved_weights_folder)

training_args_path = os.path.join(
    saved_weights_folder, "training_args.bin")

print("Training args path:", training_args_path)
training_args = torch.load(training_args_path)

# Tokenize QA data
def tokenize_qa(data):
    inputs = tokenizer(
        data["question"],
        data["context"],
        truncation=True,
        padding="max_length",
        max_length=512
    )
    start_positions = []
    end_positions = []
    for answer in data["answers"]:
        # SQuAD contains unanswerable questions so answer["answer_start"] is empty for those
        if answer["answer_start"]:
            start = answer["answer_start"][0]
            end = start + len(answer["text"][0]) 
        else:
            start = 0
            end = 0
        start_positions.append(start)
        end_positions.append(end)
    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

# Preprocess SQuAD for evaluation
dataset = valid_dataset.map(tokenize_qa, batched=True)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "start_positions", "end_positions"])
print(dataset)

# Evaluate
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset, 
    eval_dataset=valid_dataset,
    processing_class=tokenizer
)
eval_results = trainer.predict(eval_dataset=valid_dataset)
print("Evaluation Results:", eval_results)


from utils import device
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("squad_v2")
train_dataset = dataset["train"]
valid_dataset = dataset["validation"]

# Load the model and tokenizer
model_name = "huawei-noah/TinyBERT_General_4L_312D"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
model.to(device)

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

# Apply tokenization
train_dataset = train_dataset.map(tokenize_qa, batched=True)
valid_dataset = valid_dataset.map(tokenize_qa, batched=True)
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "start_positions", "end_positions"])
valid_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "start_positions", "end_positions"])

training_args = TrainingArguments(
    eval_strategy="epoch",
    logging_strategy="epoch",
    save_strategy="epoch",
    output_dir="results",
    learning_rate=3e-5,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    weight_decay=0.01,
    logging_dir="logs",
    logging_steps=50,
    report_to="wandb",
    run_name="tinyBERT_SQuAD" 
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    processing_class=tokenizer
)

trainer.train()

# Save the model weights and tokenizer
trainer.save_model('./saved_weights')
tokenizer.save_pretrained('./saved_weights')
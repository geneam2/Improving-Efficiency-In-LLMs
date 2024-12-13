from utils import preprocess_function

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    AutoModelForQuestionAnswering,
    DefaultDataCollator
)

import readline
for i in range(readline.get_current_history_length()-200,readline.get_current_history_length()):
    print (readline.get_history_item(i + 1))


def load_model(model_name):
    # model_name = "distilbert/distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    return tokenizer, model

def setup_training(args):
    tokenizer, model  = load_model(args.model_name)
    data_collator = DefaultDataCollator()
    squad = load_dataset(args.data_name)
    tokenized_squad = squad.map(lambda x: preprocess_function(x, tokenizer), batched=True, remove_columns=squad["train"].column_names)
    training_args = TrainingArguments(
        output_dir="my_awesome_qa_model",
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        push_to_hub=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_squad["train"],
        eval_dataset=tokenized_squad["test"],
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()



"""
import torch
ce = torch.nn.functional.cross_entropy

from torch.utils.data.dataloader import DataLoader
train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=data_collator, batch_size=1)
train_datapoint = next(iter(train_dataloader))
inp = {i:j.to('cuda') for i ,j in train_datapoint.items()}
outp = model(**inp)

ce(torch.stack((outp.start_logits, outp.end_logits)).squeeze(), torch.stack((inp['start_positions'], inp['end_positions'])).squeeze())
outp.loss
"""

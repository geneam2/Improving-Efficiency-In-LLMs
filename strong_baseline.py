import torch
from datasets import load_dataset
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, TrainingArguments, Trainer
import numpy as np
import collections
import time

# Change this!
device = torch.device("cpu")

"""# Fine-tune DistilBERT"""

# Load the dataset, model, and tokenizer
dataset = load_dataset("squad")
train_dataset = dataset["train"]
eval_dataset = dataset["validation"]
small_train_set = train_dataset.shuffle(seed=42).select(range(8000)) # can probably go 10x higher
small_eval_set = eval_dataset.shuffle(seed=42).select(range(2000)) # can probably go 10x higher

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
model.to(device)

model.train()

max_seq_length = 100
stride = 30 # how many tokens should overflow to next sequence https://huggingface.co/docs/transformers/en/main_classes/tokenizer#transformers.PreTrainedTokenizer.__call__.stride

# Come back to this
def preprocess_train(dataset):
    '''
        Input: dataset containing ['id', 'title', 'context', 'question', 'answers']
        This function transforms our question, context, and answers into sequences, returning
            input_ids: tokenized q and c into sequences of length max_length
            attention_mask: 1 for padding, 0 otherwise
            offset_mapping: list of tuples representing the start and end of each token
            overflow_to_sample_mapping : maps each sequence to their original id idx
        Output: dataset containing ['input_ids', 'attention_mask', 'offset_mapping', 'example_id']
    '''
    inputs = tokenizer(
        dataset["question"],
        dataset["context"],
        max_length=max_seq_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    # returns: dict_keys([ids, type_ids, tokens, offsets, attention_mask, special_tokens_mask, overflowing])

    offset_map = inputs.pop("offset_mapping")
    # cols: [[(0, 0), (0, 4), (5, 15), (16, 18), (19, 28), (29, 35), ...], ...]
    sample_map = inputs.pop("overflow_to_sample_mapping")
    # rows: [0, 0, 1, 1, 2, 2, 3, 3, ...]
    answers = dataset["answers"]

    start_positions = []
    end_positions = []
    for i, offset in enumerate(offset_map):
        sample_idx = sample_map[i]
        # Get indices of where the answers are
        answer = answers[sample_idx]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])

        # Get indices of where the context starts and ends in your input_ids
        sequence_ids = inputs.sequence_ids(i)
        # [None, 0, 0, 0, 0, 0, 0, 0, None, 1, 1, ...]
        context_start = sequence_ids.index(1)
        idx = context_start
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label is (0, 0)
        if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

train_tokenized = small_train_set.map(
    preprocess_train,
    batched=True,
    remove_columns=train_dataset.column_names,
)

eval_tokenized = small_eval_set.map(
    preprocess_train,
    batched=True,
    remove_columns=eval_dataset.column_names,
)

# Train
# Wandb.ai:
train_tokenized.set_format("torch")
eval_tokenized.set_format("torch")

training_args = TrainingArguments(
    output_dir="11/20 distilbert",
    logging_strategy="steps",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    logging_dir="logs",
    logging_steps=10,
    report_to="wandb"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=eval_tokenized,
    processing_class=tokenizer
)

start_time = time.time()
trainer.train()
training_time = time.time() - start_time

trainer.save_model('./disilbert_tuned_model2')
tokenizer.save_pretrained('./disilbert_tuned_tokenizer')

"""# Evaluate our tuned model"""

# Load the dataset, model, and tokenizer
eval_dataset = load_dataset("squad", split="validation")
small_eval_set = eval_dataset.shuffle(seed=42).select(range(200))

model_name = "./disilbert_tuned_tokenizer/"
tokenizer = AutoTokenizer.from_pretrained("./disilbert_tuned_tokenizer")
model = AutoModelForQuestionAnswering.from_pretrained("./disilbert_tuned_model")
model.to(device)

model.eval()

max_seq_length = 100
stride = 30 # how many tokens should overflow to next sequence https://huggingface.co/docs/transformers/en/main_classes/tokenizer#transformers.PreTrainedTokenizer.__call__.stride

def preprocess_eval(dataset):
    '''
        Input: dataset containing ['id', 'title', 'context', 'question', 'answers']
        This function transforms our question, context, and answers into sequences, returning
            input_ids: tokenized q and c into sequences of length max_length
            attention_mask: 1 for padding, 0 otherwise
            offset_mapping: list of tuples representing the start and end of each token
        Output: dataset containing ['input_ids', 'attention_mask', 'offset_mapping', 'example_id']
    '''
    inputs = tokenizer(
        dataset["question"],
        dataset["context"],
        max_length=max_seq_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    # returns: dict_keys(['input_ids', 'attention_mask', 'offset_mapping', 'overflow_to_sample_mapping'])

    sample_map = inputs.pop("overflow_to_sample_mapping") # maps each sequence to their original id idx
    example_ids = []
    batches = len(inputs["input_ids"])
    for i in range(batches):
        sample_idx = sample_map[i]
        example_ids.append(dataset["id"][sample_idx])

        sequence_ids = inputs.sequence_ids(i) # list showing what each position of the input_id represents, None for special tokens, 0 for question, 1 for context
        offset = inputs["offset_mapping"][i]
        # Get positions only if they're in the context
        inputs["offset_mapping"][i] = [
            end_idx if sequence_ids[start_idx] == 1 else None for start_idx, end_idx in enumerate(offset)
        ]

    inputs["example_id"] = example_ids
    return inputs

eval_tokenized = small_eval_set.map(
    preprocess_eval,
    batched=True,
    remove_columns=eval_dataset.column_names,
)

# Inference
inference_tokenized = eval_tokenized.remove_columns(["example_id", "offset_mapping"])
inference_tokenized.set_format("torch")
batch = {k: inference_tokenized[k].to(device) for k in inference_tokenized.column_names}

start_time = time.time()
with torch.no_grad():
    outputs = model(**batch)
inference_time = time.time() - start_time

# Get n best, filter out the invalid answers, and give your best answers
n_best = 10
min_answer_length = 0
max_answer_length = 30
predicted_tokens = []
predicted_answers = []

start_logits = outputs.start_logits.cpu().numpy()
end_logits = outputs.end_logits.cpu().numpy()

example_to_features = collections.defaultdict(list)
for idx, feature in enumerate(eval_tokenized):
    example_to_features[feature["example_id"]].append(idx)

for example in small_eval_set:
    id = example["id"]
    context = example["context"]
    answers = []

    for feature_index in example_to_features[id]:
        start_logit = start_logits[feature_index]
        end_logit = end_logits[feature_index]
        offsets = eval_tokenized["offset_mapping"][feature_index]

        start_idx_sorted = np.argsort(start_logit)[::-1]
        start_indexes = start_idx_sorted[:n_best]

        end_idx_sorted = np.argsort(end_logit)[::-1]
        end_indexes = end_idx_sorted[:n_best]

        for start_index in start_indexes:
            for end_index in end_indexes:
                # Skip answers that are not fully in the context
                if offsets[start_index] is None or offsets[end_index] is None:
                    continue
                # Skip answers with a length < 0 or > max_answer_length.
                answer_length = end_index - start_index + 1
                if (answer_length < min_answer_length or answer_length > max_answer_length):
                    continue

                answers.append({"text": context[offsets[start_index][0] : offsets[end_index][1]],
                                "logit_score": start_logit[start_index] + end_logit[end_index]
                                })

    best_answer = max(answers, key=lambda x: x["logit_score"])
    predicted_answers.append({"id": id, "prediction_text": best_answer["text"]})

# Get the tokenized versions of the predicted and actual
predicted_tokenized = []
for example in predicted_answers:
    predicted = {}
    predicted['id'] = example["id"]
    predicted['prediction_text'] = example["prediction_text"]
    predicted['prediction_tokenized'] = tokenizer(example["prediction_text"])['input_ids']
    predicted_tokenized.append(predicted)

actual_tokenized = []
for example in small_eval_set:
    actual = {}
    actual['id'] = example["id"]
    actual_answers = example["answers"]["text"]
    actual['actual_text'] = actual_answers
    actual['actual_tokenized'] = tokenizer(actual_answers)["input_ids"]
    actual_tokenized.append(actual)

example_predicted = predicted_tokenized[0]
example_actual = actual_tokenized[0]
print("Predicted: ", example_predicted)
print("Actual: ", example_actual)

# Save dictionary to a JSON file
import json

with open("strong_predicted_output.json", "w", encoding="utf-8") as f:
    json.dump(predicted_tokenized, f, indent=4)

with open("strong_actual_output.json", "w", encoding="utf-8") as f:
    json.dump(actual_tokenized, f, indent=4)

print(f"{round(training_time, 2)} seconds")
print(f"{round(inference_time, 2)} seconds")
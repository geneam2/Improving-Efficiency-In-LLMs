import torch
from datasets import load_dataset
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import numpy as np
import collections
import time

device = torch.set_default_device("mps")
# device = "cuda" if torch.cuda.is_available() else "cpu"
device

# Load the dataset, model, and tokenizer
eval_dataset = load_dataset("squad", split="validation")
small_eval_set = eval_dataset.shuffle(seed=42).select(range(200))

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
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

with open("simple_predicted_output.json", "w", encoding="utf-8") as f:
    json.dump(predicted_tokenized, f, indent=4)

with open("simple_actual_output.json", "w", encoding="utf-8") as f:
    json.dump(actual_tokenized, f, indent=4)

print(f"{round(inference_time, 2)} seconds")
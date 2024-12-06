import collections
from time import time

import torch
import numpy as np
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)

from utils import register_to, TASK_REGISTRY

class TaskClass:
    def __init__(self, model):
        self.model = model
    
    def train(self, args):
        raise NotImplementedError

@register_to(TASK_REGISTRY)
class SQuADV2(TaskClass):
    
    def train(self, args):
        training_args = TrainingArguments(
            **args.__dict__

        )
        trainer = Trainer(
            model=self.model.model,
            args=training_args,
            train_dataset=self.model.train_tokenized,
            eval_dataset=self.model.eval_tokenized,
            processing_class=self.model.tokenizer,
        )
        start_time = time.time()
        trainer.train()
        training_time = time.time() - start_time
        print(f"Training Time: {round(training_time, 2)} seconds")
        trainer.save_model('./disilbert_tuned_model2')
        self.model.save_pretrained('./disilbert_tuned_tokenizer')

    def evaluate(self, args):
        model = self.model.model        
        tokenizer = self.model.tokenizer

        small_eval_set = self.model.small_eval_set
        eval_tokenized = self.model.eval_tokenized
        inference_tokenized = self.model.inference_tokenized

        batch = {k: inference_tokenized[k].to(self.model.device) for k in inference_tokenized.column_names}

        model.eval()

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

        print(f"Inference Time: {round(inference_time, 2)} seconds")

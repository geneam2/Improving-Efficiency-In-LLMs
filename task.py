

import torch
from torch.utils.data.dataloader import DataLoader

from evaluate import load
from datasets import load_dataset
from transformers import DataCollatorWithPadding
from transformers import (
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
    AutoTokenizer
)

from utils import register_to, TASK_REGISTRY
from qa_utils import postprocess_qa_predictions

# GENE ADDED
from functools import partial
from transformers.data.metrics.squad_metrics import compute_exact, compute_f1, make_eval_dict

class TaskClass:

    def __init__(self, task_args, train_args, model_fn):
        self.tokenizer = AutoTokenizer.from_pretrained(task_args.model_name)
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        self.init_model(model_fn, task_args)
        self.train_args = train_args
        self.task_args = task_args

    def init_model(self):
        raise NotImplementedError

    @staticmethod
    def process_function(examples, tokenizer, input_fields):
        raise NotImplementedError

    def loss_function(self, hypo, targ):
        raise NotImplementedError
    
    def prepare(self):
        raise NotImplementedError
    
    def evaluate(self):
        raise NotImplementedError

    def extract_answer_from_output(self, _, outp):
        raise NotImplementedError

    def extract_label_from_input(self, inp):
        raise NotImplementedError

    def compute_metric(self, preds, labels):
        raise NotImplementedError

    def print_model_params(self):
        trainable_params = 0
        total_params = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                trainable_params += param.numel()
            total_params += param.numel()
        print(f"Total params {total_params} | Trainable params {trainable_params} ({trainable_params/total_params})")

@register_to(TASK_REGISTRY)
class SQuADv2(TaskClass):

    def __init__(self, task_args, train_args, model_fn):
        super().__init__(task_args, train_args, model_fn)
        self.criterion = torch.nn.functional.cross_entropy
        self.metric = load("squad")

    @staticmethod
    def process_function(examples, tokenizer, max_seq_len=384):
        # questions = [q.strip() for q in examples["question"]]
        
        # inputs = tokenizer(
        #     questions,
        #     examples["context"],
        #     max_length=max_seq_len,
        #     truncation=True if max_seq_len <= 512 else "only_second",
        #     return_offsets_mapping=True,
        #     padding="max_length",
        # )

        # offset_mapping = inputs.pop("offset_mapping")
        # answers = examples["answers"]
        # start_positions = []
        # end_positions = []

        # for i, offset in enumerate(offset_mapping):
        #     answer = answers[i]
        #     if len(answer["answer_start"]) == 0:
        #         start_positions.append(0)
        #         end_positions.append(0)
        #     else:
        #         start_char = answer["answer_start"][0]
        #         end_char = answer["answer_start"][0] + len(answer["text"][0])
        #         sequence_ids = inputs.sequence_ids(i)
        #         # Find the start and end of the context
        #         idx = 0
        #         while sequence_ids[idx] != 1:
        #             idx += 1
        #         context_start = idx
        #         while sequence_ids[idx] == 1:
        #             idx += 1
        #         context_end = idx - 1
        #         # If the answer is not fully inside the context, label it (0, 0)
        #         if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
        #             start_positions.append(0)
        #             end_positions.append(0)
        #         else:
        #             # Otherwise it's the start and end token positions
        #             idx = context_start
        #             while idx <= context_end and offset[idx][0] <= start_char:
        #                 idx += 1
        #             start_positions.append(idx - 1)
        #             idx = context_end
        #             while idx >= context_start and offset[idx][1] >= end_char:
        #                 idx -= 1
        #             end_positions.append(idx + 1)
        inputs = tokenizer(
            examples["question"],
            examples["context"],
            max_length=max_seq_len,
            truncation="only_second",
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )
        # returns: dict_keys([ids, type_ids, tokens, offsets, attention_mask, special_tokens_mask, overflowing])

        offset_map = inputs.pop("offset_mapping")
        # cols: [[(0, 0), (0, 4), (5, 15), (16, 18), (19, 28), (29, 35), ...], ...]
        sample_map = inputs.pop("overflow_to_sample_mapping")
        # rows: [0, 0, 1, 1, 2, 2, 3, 3, ...]
        answers = examples["answers"]

        start_positions = []
        end_positions = []
        for i, offset in enumerate(offset_map):
            sample_idx = sample_map[i]
            # Get indices of where the answers are
            answer = answers[sample_idx]
            # No answer
            if len(answer["answer_start"]) == 0 or len(answer["text"]) == 0:
                start_positions.append(0)
                end_positions.append(0)
                continue
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
        
        # Verify our answers are good
        # answers = [tokenizer.decode(inputs["input_ids"][i][start_positions[i]:end_positions[i]+1], skip_special_tokens=True).strip() for i in range(len(inputs["input_ids"]))]

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs

    @staticmethod
    def process_helper(examples, tokenizer, max_seq_len=384):
        inputs = tokenizer(
            examples["question"],
            examples["context"],
            max_length=max_seq_len,
            truncation="only_second",
            stride=30,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        sample_map = inputs.pop("overflow_to_sample_mapping") # maps each sequence to their original id idx
        example_ids = []
        batches = len(inputs["input_ids"])
        for i in range(batches):
            sample_idx = sample_map[i]
            example_ids.append(examples["id"][sample_idx])

            sequence_ids = inputs.sequence_ids(i) # list showing what each position of the input_id represents, None for special tokens, 0 for question, 1 for context
            offset = inputs["offset_mapping"][i]
            # Get positions only if they're in the context
            inputs["offset_mapping"][i] = [
                o if sequence_ids[k] == 1 else (0, 0) for k, o in enumerate(offset)
            ]

        del inputs["offset_mapping"]
        inputs["example_id"] = example_ids
        return inputs

    def init_model(self, model_fn, task_args):
        self.model = model_fn(task_args.model_name)

    def prepare(self):
        squad = load_dataset("rajpurkar/squad_v2")
        tokenized_squad = squad.map(
            lambda x: self.process_function(x,
                        tokenizer=self.tokenizer, 
                        max_seq_len=self.train_args.max_seq_len
                    ),
            batched=True,
            remove_columns=squad["train"].column_names,
        )
        train_dataloader = DataLoader(
            tokenized_squad['train'],
            shuffle=True,
            collate_fn=self.data_collator,
            batch_size=self.train_args.train_batch,
        )
        validation_dataloader = DataLoader(
            tokenized_squad['validation'],
            shuffle=False,
            collate_fn=self.data_collator,
            batch_size=self.train_args.val_batch,
        )

        tokenized_helper = squad["validation"].map(
            lambda x: self.process_helper(x,
                        tokenizer=self.tokenizer, 
                        max_seq_len=self.train_args.max_seq_len
                    ),
            batched=True,
            remove_columns=squad["validation"].column_names,
        )

        helper_dataloader = DataLoader(
            tokenized_helper,
            shuffle=False,
            # collate_fn=self.data_collator,
            batch_size=self.train_args.val_batch,
        )
        
        return (
            train_dataloader,
            validation_dataloader,
            helper_dataloader,
        )

    def loss_function(self, hypo, targ):
        # hypo.shape == (bsz, 2, seq_len)
        # targ.shape == (bsz)
        return hypo.loss

    def extract_answer_from_output(self, outp, inp, helper):
        logit_ans = torch.stack([
            outp.start_logits.argmax(dim=-1),
            outp.end_logits.argmax(dim=-1),
        ], dim=1).tolist()

        decoded_answers = []
        for i, indices in enumerate(logit_ans):
            start_index, end_index = indices

            # Handle invalid spans
            if end_index < start_index:
                answer = ""
            else:
                # THIS IS THE REASON F1 score is low
                # Map tokens back to original context
                # offsets = helper['offset_mapping'][i]
                # if offsets[0][start_index] and offsets[1][end_index]:
                #     start_char, end_char = offsets[start_index][0], offsets[end_index][1]
                #     answer = inp['context'][i][start_char:end_char]
                # else:
                #     answer = ""
                answer_tokens = inp['input_ids'][i][start_index: end_index+1]
                answer = self.tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()

            decoded_answers.append({"id": helper["example_id"][i], "prediction_text": answer})
        return decoded_answers


    def extract_label_from_input(self, inp, helper):
        # Extracts the actual start and end logits from the input
        num_answers = len(inp["input_ids"])
        label_ans = []
        for i in range(num_answers):
            start_index, end_index = inp["start_positions"][i], inp["end_positions"][i]

            if end_index < start_index:
                answer = ""
            else:
                answer_tokens = inp['input_ids'][i][start_index: end_index+1]
                answer = self.tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()
            
            label_ans.append({
                            "id": helper["example_id"][i],
                            "answers": {'answer_start': [start_index.item()], 'text': [answer]}
                            })
            
        return label_ans

    def compute_metric(self, preds, labels):
        return self.metric.compute(
            predictions=preds,
            references=labels,
        )
        
    def inference(self, inp):
        outp = self.model(**inp)
        return self.extract_answer_from_output(outp)

    def evaluate(self, inp, label):
        pred = self.inference(inp)
        return pred, label.detach().tolist()
        

class SequenceClassification(TaskClass):

    def __init__(self, task_args, train_args, model_fn):
        super().__init__(task_args, train_args, model_fn)
        self.criterion = torch.nn.functional.cross_entropy
        self.metric = load("glue", self.task_args.task_name.lower())

    def init_model(self, model_fn, task_args):
        if getattr(task_args, "lora_r", None) is None:
            self.model = model_fn(task_args.model_name, num_labels=2)
        else:
            self.model = model_fn(
                task_args.model_name,
                lora_r=task_args.lora_r,
                lora_alpha=task_args.lora_alpha,
                num_labels=2
            )

    @staticmethod
    def process_function(examples, tokenizer, input_fields, max_seq_len=None):
        max_seq_len = 384 if max_seq_len is None else max_seq_len
        if len(input_fields) == 1:
            inp = tokenizer(
                [i.strip() for i in examples[input_fields[0]]],
                max_length=max_seq_len,
                truncation=True,
            )
        else:
            inp = tokenizer(
                [i.strip() for i in examples[input_fields[0]]],
                [i.strip() for i in examples[input_fields[1]]],
                max_length=max_seq_len,
                truncation=True,   
            )
        inp["label"] = examples["label"]
        return inp

    def loss_function(self, hypo, targ):
        # hypo.shape == (bsz, num_classes)
        # targ.shape == (bsz)
        return hypo.loss

    def extract_answer_from_output(self, _, outp):
        return outp.logits.argmax(dim=1).detach().tolist()

    def extract_label_from_input(self, inp):
        return inp['labels'].detach().tolist()

    def inference(self, inp):
        outp = self.model(**inp)
        return self.extract_answer_from_output(_, outp)

    def compute_metric(self, preds, labels):
        return self.metric.compute(
            predictions=preds,
            references=labels,
        )

    def evaluate(self, inp, label):
        pred = self.inference(inp)
        return pred, label.detach().tolist()

@register_to(TASK_REGISTRY)
class MNLI(SequenceClassification):

    def init_model(self, model_fn, task_args):
        if getattr(task_args, "lora_r") is None:
            self.model = model_fn(task_args.model_name, num_labels=3)
        else:
            self.model = model_fn(
                task_args.model_name,
                lora_r=task_args.lora_r,
                lora_alpha=task_args.lora_alpha,
                num_labels=3
            )

    def prepare_eval(self):
        ds = load_dataset("nyu-mll/glue", self.task_args.task_name.lower(), split="test_matched")
        tokenized_ds = ds.map(
            lambda x: self.process_function(
            x, self.tokenizer, ["premise", "hypothesis"], getattr(self.train_args, "max_seq_len", None)),
            batched=True,
            remove_columns=ds.column_names,
        )
        test_matched_dataloader = DataLoader(
            tokenized_ds,
            shuffle=False,
            collate_fn=self.data_collator,
            batch_size=self.train_args.test_batch,
        )
        return test_matched_dataloader

    def prepare(self):
        ds = load_dataset("nyu-mll/glue", self.task_args.task_name.lower())
        tokenized_ds = ds.map(
            lambda x: self.process_function(
                x, self.tokenizer, ["premise", "hypothesis"], getattr(self.train_args, "max_seq_len", None)),
            batched=True,
            remove_columns=ds["train"].column_names,
        )
        train_dataloader = DataLoader(
            tokenized_ds['train'],
            shuffle=True,
            collate_fn=self.data_collator,
            batch_size=self.train_args.train_batch,
        )
        validation_matched_dataloader = DataLoader(
            tokenized_ds['validation_matched'],
            shuffle=False,
            collate_fn=self.data_collator,
            batch_size=self.train_args.val_batch,
        )
        test_matched_dataloader = DataLoader(
            tokenized_ds['test_matched'],
            shuffle=False,
            collate_fn=self.data_collator,
            batch_size=self.train_args.test_batch,
        )
        # "train", "validation_matched", "test_matched"
        # ['premise', 'hypothesis', 'label', 'idx']
        # task: SequenceClassification
        # label: 0, 1, 2
        return (
            train_dataloader,
            validation_matched_dataloader,
            test_matched_dataloader,
        )

@register_to(TASK_REGISTRY)
class SST2(SequenceClassification):

    def prepare(self):
        ds = load_dataset("nyu-mll/glue", self.task_args.task_name.lower())
        tokenized_ds = ds.map(
            lambda x: self.process_function(
                x, self.tokenizer, ["sentence"], getattr(self.train_args, "max_seq_len", None)),
            batched=True,
            remove_columns=ds["train"].column_names,
        )
        train_dataloader = DataLoader(
            tokenized_ds['train'],
            shuffle=True,
            collate_fn=self.data_collator,
            batch_size=self.train_args.train_batch,
        )
        validation_dataloader = DataLoader(
            tokenized_ds['validation'],
            shuffle=False,
            collate_fn=self.data_collator,
            batch_size=self.train_args.val_batch,
        )
        test_dataloader = DataLoader(
            tokenized_ds['test'],
            shuffle=False,
            collate_fn=self.data_collator,
            batch_size=self.train_args.test_batch,
        )
        # "train", "validation", "test"
        # ['sentence', 'label', 'idx']
        # task: SequenceClassification
        # label: 0, 1
        # stanford sentiment treebank (sst2) tests for sentiment (pos/neg) of given sentence

        return (
            train_dataloader,
            validation_dataloader,
            test_dataloader,
        )

@register_to(TASK_REGISTRY)
class MRPC(SequenceClassification):

    def prepare(self):
        ds = load_dataset("nyu-mll/glue", self.task_args.task_name.lower())
        tokenized_ds = ds.map(
            lambda x: self.process_function(
                x, self.tokenizer, ['sentence1', 'sentence2'], getattr(self.train_args, "max_seq_len", None)),
            batched=True,
            remove_columns=ds["train"].column_names,
        )
        train_dataloader = DataLoader(
            tokenized_ds['train'],
            shuffle=True,
            collate_fn=self.data_collator,
            batch_size=self.train_args.train_batch,
        )
        validation_dataloader = DataLoader(
            tokenized_ds['validation'],
            shuffle=False,
            collate_fn=self.data_collator,
            batch_size=self.train_args.val_batch,
        )
        test_dataloader = DataLoader(
            tokenized_ds['test'],
            shuffle=False,
            collate_fn=self.data_collator,
            batch_size=self.train_args.test_batch,
        )
        # "train", "validation", "test"
        # ['sentence1', 'sentence2', 'label', 'idx']
        # task: SequenceClassification
        # label: 0, 1
        # microsoft research paraphrase corpus (mrpc)mtests for semantic equivalence 

        return (
            train_dataloader,
            validation_dataloader,
            test_dataloader,
        )

@register_to(TASK_REGISTRY)
class CoLA(SequenceClassification):


    def prepare(self):
        ds = load_dataset("nyu-mll/glue", self.task_args.task_name.lower())
        tokenized_ds = ds.map(
            lambda x: self.process_function(
                x, self.tokenizer, ['sentence'], getattr(self.train_args, "max_seq_len", None)),
            batched=True,
            remove_columns=ds["train"].column_names,
        )
        train_dataloader = DataLoader(
            tokenized_ds['train'],
            shuffle=True,
            collate_fn=self.data_collator,
            batch_size=self.train_args.train_batch,
        )
        validation_dataloader = DataLoader(
            tokenized_ds['validation'],
            shuffle=False,
            collate_fn=self.data_collator,
            batch_size=self.train_args.val_batch,
        )
        test_dataloader = DataLoader(
            tokenized_ds['test'],
            shuffle=False,
            collate_fn=self.data_collator,
            batch_size=self.train_args.test_batch,
        )
        # "train", "validation", "test"
        # ['sentence', 'label', 'idx']
        # task: SequenceClassification
        # label: 0, 1
        # tests whether the given sentence is grammatically correct english

        return (
            train_dataloader,
            validation_dataloader,
            test_dataloader,
        )

@register_to(TASK_REGISTRY)
class QNLI(SequenceClassification):


    def prepare(self):
        ds = load_dataset("nyu-mll/glue", self.task_args.task_name.lower())
        tokenized_ds = ds.map(
            lambda x: self.process_function(
                x, self.tokenizer, ['question', 'sentence'], getattr(self.train_args, "max_seq_len", None)),
            batched=True,
            remove_columns=ds["train"].column_names,
        )
        train_dataloader = DataLoader(
            tokenized_ds['train'],
            shuffle=True,
            collate_fn=self.data_collator,
            batch_size=self.train_args.train_batch,
        )
        validation_dataloader = DataLoader(
            tokenized_ds['validation'],
            shuffle=False,
            collate_fn=self.data_collator,
            batch_size=self.train_args.val_batch,
        )
        test_dataloader = DataLoader(
            tokenized_ds['test'],
            shuffle=False,
            collate_fn=self.data_collator,
            batch_size=self.train_args.test_batch,
        )
        # "train", "validation", "test"
        # ['question', 'sentence', 'label', 'idx']
        # task: SequenceClassification
        # label: 0, 1
        # tests for whether the answer to the question can be found in the question
        return (
            train_dataloader,
            validation_dataloader,
            test_dataloader,
        )

@register_to(TASK_REGISTRY)
class QQP(SequenceClassification):


    def prepare(self):
        ds = load_dataset("nyu-mll/glue", self.task_args.task_name.lower())
        tokenized_ds = ds.map(
            lambda x: self.process_function(
                x, self.tokenizer, ['question1', 'question2'], getattr(self.train_args, "max_seq_len", None)),
            batched=True,
            remove_columns=ds["train"].column_names,
        )
        train_dataloader = DataLoader(
            tokenized_ds['train'],
            shuffle=True,
            collate_fn=self.data_collator,
            batch_size=self.train_args.train_batch,
        )
        validation_dataloader = DataLoader(
            tokenized_ds['validation'],
            shuffle=False,
            collate_fn=self.data_collator,
            batch_size=self.train_args.val_batch,
        )
        test_dataloader = DataLoader(
            tokenized_ds['test'],
            shuffle=False,
            collate_fn=self.data_collator,
            batch_size=self.train_args.test_batch,
        )
        # "train", "validation", "test"
        # ['question1', 'question2', 'label', 'idx']
        # task: SequenceClassification
        # label: 0, 1
        # quora question pairs (qqp) tests for semantic equivalence 
        return (
            train_dataloader,
            validation_dataloader,
            test_dataloader,
        )

@register_to(TASK_REGISTRY)
class RTE(SequenceClassification):


    def prepare(self):
        ds = load_dataset("nyu-mll/glue", self.task_args.task_name.lower())
        tokenized_ds = ds.map(
            lambda x: self.process_function(
                x, self.tokenizer, ['sentence1', 'sentence2'], getattr(self.train_args, "max_seq_len", None)),
            batched=True,
            remove_columns=ds["train"].column_names,
        )
        train_dataloader = DataLoader(
            tokenized_ds['train'],
            shuffle=True,
            collate_fn=self.data_collator,
            batch_size=self.train_args.train_batch,
        )
        validation_dataloader = DataLoader(
            tokenized_ds['validation'],
            shuffle=False,
            collate_fn=self.data_collator,
            batch_size=self.train_args.val_batch,
        )
        test_dataloader = DataLoader(
            tokenized_ds['test'],
            shuffle=False,
            collate_fn=self.data_collator,
            batch_size=self.train_args.test_batch,
        )
        # "train", "validation", "test"
        # ['sentence1', 'sentence2', 'label', 'idx']
        # task: SequenceClassification
        # label: 0, 1
        # recognizing textual entailment (rte) tests textual entailment (collapses neutral & contradiction into not entailment)
        return (
            train_dataloader,
            validation_dataloader,
            test_dataloader,
        )

@register_to(TASK_REGISTRY)
class STSB(SequenceClassification):

    def prepare(self):
        ds = load_dataset("nyu-mll/glue", self.task_args.task_name.lower())
        tokenized_ds = ds.map(
            lambda x: self.process_function(
                x, self.tokenizer, ['sentence1', 'sentence2'], getattr(self.train_args, "max_seq_len", None)),
            batched=True,
            remove_columns=ds["train"].column_names,
        )
        train_dataloader = DataLoader(
            tokenized_ds['train'],
            shuffle=True,
            collate_fn=self.data_collator,
            batch_size=self.train_args.train_batch,
        )
        validation_dataloader = DataLoader(
            tokenized_ds['validation'],
            shuffle=False,
            collate_fn=self.data_collator,
            batch_size=self.train_args.val_batch,
        )
        test_dataloader = DataLoader(
            tokenized_ds['test'],
            shuffle=False,
            collate_fn=self.data_collator,
            batch_size=self.train_args.test_batch,
        )
        # "train", "validation", "test"
        # ['sentence1', 'sentence2', 'label', 'idx']
        # task: SequenceClassification
        # label: floating point from 0 to 5
        # pair is human-annotated with a similarity score from 1 to 5
        return (
            train_dataloader,
            validation_dataloader,
            test_dataloader,
        )

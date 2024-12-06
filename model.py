from datasets import load_dataset
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

from utils import register_to, MODEL_REGISTRY

class ModelClass:

    def __init__(self, model_args, data_args):
        self.init_model(model_args)
        self.init_dataset(data_args)

@register_to(MODEL_REGISTRY)
class StrongBaseline(ModelClass):

    def init_model(self, args):
        self.model = AutoModelForQuestionAnswering.from_pretrained(args.model_name)
        self.model.to(args.device)
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def init_data(self, args):

        def preprocess_train(dataset, tokenizer=self.tokenizer):
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
                max_length=args.max_seq_length,
                truncation="only_second",
                stride=args.stride,
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


        def preprocess_eval(dataset, tokenizer=self.tokenizer):
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
                max_length=args.max_seq_length,
                truncation="only_second",
                stride=args.stride,
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

        dataset = load_dataset("squad")
        train_dataset = dataset["train"]
        val_dataset = dataset["validation"]

        small_train_set = train_dataset.shuffle(seed=42).select(range(8000)) # can probably go 10x higher
        small_val_set = val_dataset.shuffle(seed=42).select(range(2000)) # can probably go 10x higher

        self.train_tokenized = small_train_set.map(
            preprocess_train,
            batched=True,
            remove_columns=train_dataset.column_names,
        )

        self.val_tokenized = small_val_set.map(
            preprocess_train,
            batched=True,
            remove_columns=val_dataset.column_names,
        )

        eval_dataset = load_dataset("squad", split="validation")
        self.small_eval_set = eval_dataset.shuffle(seed=42).select(range(200))
        self.eval_tokenized = self.small_eval_set.map(
            preprocess_eval,
            batched=True,
            remove_columns=eval_dataset.column_names,
        )

        # Inference
        self.inference_tokenized = self.eval_tokenized.remove_columns(["example_id", "offset_mapping"])
        self.inference_tokenized.set_format("torch")


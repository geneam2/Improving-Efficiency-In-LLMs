from tqdm import tqdm

import torch

class FakeWandB:

    def __init__(self):
        self.logs = []

    def log(self, logs):
        self.logs.append(logs)

class CustomEvaluator:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def __init__(self, task):
        self.task = task

    def prepare_eval(self, args):
        test_dl = self.task.prepare_eval()        
        self.task.model = self.task.model.to(self.device)
        if not args.from_hf:
            checkpoint = torch.load(args.checkpoint, map_location=self.device)
            self.task.model.load_state_dict(checkpoint['model_state_dict'])
        return test_dl

    def evaluate(self, args):
        self.task.model.eval()
        test_dl = self.prepare_eval(args)

        self.task.print_model_params()
        model = self.task.model.to(self.device)
        # ========== evaluation ==========
        preds = []
        labels = []
        with torch.no_grad():
            for step, batch in enumerate(tqdm(test_dl)):
                # ========== forward pass ==========
                batch = {i:j.to(self.device) for i,j in batch.items()}
                outputs = model(**batch)

                # ========== compute metric ==========
                preds.extend(
                    self.task.extract_answer_from_output(outputs)
                )
                labels.extend(
                    self.task.extract_label_from_input(batch)
                )

        val_result = self.task.compute_metric(preds, labels)
        print("Test set acc: {}".format(val_result))
        return val_result

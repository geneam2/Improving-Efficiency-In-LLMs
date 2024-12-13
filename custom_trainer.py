import wandb
import torch
from accelerate import Accelerator
from tqdm import tqdm

class FakeWandB:

    def __init__(self):
        self.logs = []

    def log(self, logs):
        self.logs.append(logs)

class CustomTrainer:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def __init__(self, task, wandb_config):
        self.task = task
        if wandb_config is not None:
            wandb.login(key=wandb_config.api_key)
            wandb.init(
                # Set the project where this run will be logged
                project=wandb_config.project_name,
                name=wandb_config.experiment_name,
                # Track hyperparameters and run metadata
                config={
                    "task":self.task.task_args.__dict__,
                    "train_args":self.task.train_args.__dict__,
                }
            )
            self.wandb = wandb
        else:
            self.wandb = FakeWandB()

    def prepare_train(self, args):
        
        train_dl, val_dl, test_dl = self.task.prepare()
        total_training_steps = len(train_dl) * args.epochs

        self.optim = torch.optim.AdamW(
            self.task.model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optim,
            start_factor=1e-10,
            total_iters=total_training_steps*args.warmup_ratio
        )

        self.task.model = self.task.model.to(self.device)

        return train_dl, val_dl, test_dl

    def train(self, args):
        self.task.model.train()
        device = self.device
        train_dl, val_dl, test_dl = self.prepare_train(args)
        accelerator = Accelerator(gradient_accumulation_steps=args.grad_accum)
        model, self.optim, train_dl, self.scheduler = accelerator.prepare(
            self.task.model, self.optim, train_dl, self.scheduler
        )

        steps_per_epoch = len(train_dl)
        val_steps_per_epoch = len(val_dl)
        for epoch in range(args.epochs):

            # ========== training ==========
            losses = []
            num_datapoints = 0

            for step, batch in enumerate(train_dl):
                # breakpoint()
                with accelerator.accumulate(model):
                    # ========== forward pass ==========
                    batch = {i:j.to(device) for i,j in batch.items()}
                    outputs = model(**batch)
                    loss = self.task.loss_function(outputs.logits, batch['labels'])

                    # ========== backpropagation ==========
                    accelerator.backward(loss)
                    self.optim.step()
                    self.scheduler.step()
                    self.optim.zero_grad()

                    # ========== logging ==========
                    loss_for_logging = loss.detach().tolist()
                    losses.append(loss_for_logging*len(batch['labels']))
                    num_datapoints += len(batch['labels'])
                    self.wandb.log({
                        "train/loss": loss_for_logging, 
                        "train/learning_rate": self.scheduler.get_last_lr()[0]
                    })
                    print("Epoch {} training loss: {}".format(
                        step/steps_per_epoch, loss_for_logging), end="\r")

            print("\nEpoch {} avg training loss: {}".format(
                epoch, sum(losses)/num_datapoints))

            # ========== validation ==========
            val_losses = []
            num_datapoints = 0
            preds = []
            labels = []
            for step, batch in enumerate(val_dl):
                # ========== forward pass ==========
                batch = {i:j.to(self.device) for i,j in batch.items()}
                outputs = model(**batch)
                loss = self.task.loss_function(outputs.logits, batch['labels'])

                # ========== compute metric ==========
                preds.extend(
                    self.task.extract_answer_from_output(outputs)
                )
                labels.extend(
                    batch['labels'].detach().tolist()
                )

                # ========== logging ==========
                val_loss_for_logging = loss.detach().tolist()
                val_losses.append(val_loss_for_logging*len(batch['labels']))
                num_datapoints += len(batch['labels'])
                print("Epoch {} validation loss: {}".format(
                    step/val_steps_per_epoch, val_loss_for_logging), end="\r")

            self.wandb.log({"val/loss": sum(val_losses)/num_datapoints})
            print("Epoch {} avg validation loss: {}".format(
                    epoch, sum(val_losses)/num_datapoints))
            val_result = self.task.metric.compute(
                predictions=preds,
                references=labels,
            )
            print("Epoch {} validation acc: {}".format(
                epoch, val_result), end="\r")
            self.wandb.log({"val/{}".format(i):j for i,j in val_result.items()})

    def evaluate(self, dl):
        pred_list = []
        label_list = []
        with torch.no_grad():
            for inp in dl:
                preds, labels = self.task.evaluate(inp, inp['label'])
                pred_list.extend(preds)
                label_list.extend(labels)
    
        result = self.task.metric.compute(
            predictions=pred_list,
            references=label_list,
        )

        return result

    def inference(self, dl):
        self.task.model.eval()
        infer_list = []
        with torch.no_grad():
            for inp in dl:
                preds = self.task.inference(inp)
                infer_list.extend(preds)
        return infer_list

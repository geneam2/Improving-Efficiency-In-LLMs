import wandb
import torch
from accelerate import Accelerator
from tqdm import tqdm
# For checkpointing
import os
import glob
import re

from custom_scheduler import InverseSqrtScheduler

class FakeWandB:

    def __init__(self):
        self.logs = []

    def log(self, logs):
        self.logs.append(logs)

class CustomTrainer:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def __init__(self, task, wandb_config, sweep=False):
        self.task = task
        self.resume_from_checkpoint = wandb_config.resume_from_checkpoint
        if sweep:
            self.wandb = wandb
        else:
            if wandb_config is not None:
                wandb.login(key=wandb_config.api_key)
                if not wandb_config.resume_from_checkpoint:
                    wandb.init(
                        project=wandb_config.project_name,
                        name=wandb_config.experiment_name,
                        config={
                            "task": self.task.task_args.__dict__,
                            "train_args": self.task.train_args.__dict__,
                        }
                    )
                else:
                    wandb.init(
                        # Set the project where this run will be logged
                        project=wandb_config.project_name,
                        name=wandb_config.experiment_name,
                        # For checkpointing
                        id=wandb_config.existing_run_id,
                        resume="allow",
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
        if getattr(args, "scheduler", None) is None:
            self.scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optim,
                start_factor=1e-10,
                total_iters=total_training_steps*args.warmup_ratio
            )
        elif getattr(args, "scheduler", None) == "InverseSqrt":
            self.scheduler = InverseSqrtScheduler(
                optimizer=self.optim,
                warmup_updates=total_training_steps*args.warmup_ratio,
                warmup_init_lr=1e-10,
                lr=args.learning_rate,
            )

        self.task.model = self.task.model.to(self.device)

        return train_dl, val_dl, test_dl

    def save_checkpoint(self, checkpoint_path, epoch, step, model, optimizer, scheduler):
        os.makedirs(checkpoint_path, exist_ok=True)
        checkpoint_file = os.path.join(checkpoint_path, f"epoch_{epoch}_step_{step}.pt")

        torch.save({
            'epoch': epoch,
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()
        }, checkpoint_file)

    def load_checkpoint(self, checkpoint_path, model, optimizer, scheduler):
        # Identify the latest file
        checkpoint_files = glob.glob(os.path.join(checkpoint_path, "epoch_*_step_*.pt"))
        if not checkpoint_files:
            return 0, 0
        
        latest_epoch = -1
        latest_step = -1
        latest_file = None
        for f in checkpoint_files:
            match = re.search(r'epoch_(\d+)_step_(\d+)\.pt', f)
            if match:
                epoch = int(match.group(1))
                step = int(match.group(2))
                # Compare to find the latest checkpoint
                if epoch > latest_epoch or (epoch == latest_epoch and step > latest_step):
                    latest_epoch = epoch
                    latest_step = step
                    latest_file = f

        if latest_file is None:
            return 0, 0

        # Load everything
        checkpoint = torch.load(latest_file, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint['epoch'], checkpoint['step']

    def train(self, args):
        self.task.model.train()
        device = self.device
        # Changed test_dl to tokenized_validation_helper
        train_dl, val_dl, tokenized_val_helper = self.prepare_train(args)
        accelerator = Accelerator(gradient_accumulation_steps=args.grad_accum)
        model, self.optim, train_dl, self.scheduler = accelerator.prepare(
            self.task.model, self.optim, train_dl, self.scheduler
        )
        self.task.print_model_params()
        steps_per_epoch = len(train_dl)
        val_steps_per_epoch = len(val_dl)

        # ========== load checkpoints ==========
        start_epoch, current_step = 0, 0
        if args.checkpoint_path and self.resume_from_checkpoint and os.path.exists(args.checkpoint_path):
            start_epoch, current_step = self.load_checkpoint(args.checkpoint_path, model, self.optim, self.scheduler)
        
        try:
            for epoch in range(start_epoch, args.epochs):

                # ========== training ==========
                losses = []
                num_datapoints = 0

                '''
                if current_step > 0:
                    train_dl_iter = iter(train_dl)
                    for _ in range(current_step):
                        next(train_dl_iter)
                else:
                    train_dl_iter = iter(train_dl)
                
                for batch in train_dl_iter:

                    with accelerator.accumulate(model):

                        # ========== forward pass ==========
                        batch = {i:j.to(device) for i,j in batch.items()}
                        outputs = model(**batch)
                        loss = self.task.loss_function(outputs, batch)

                        # ========== backpropagation ==========
                        accelerator.backward(loss)
                        self.optim.step()
                        self.scheduler.step()
                        self.optim.zero_grad()

                        # ========== logging ==========
                        loss_for_logging = loss.detach().tolist()
                        losses.append(loss_for_logging*len(batch))
                        num_datapoints += len(batch)
                        self.wandb.log({
                            "train/loss": loss_for_logging, 
                            "train/learning_rate": self.scheduler.get_last_lr()[0]
                        }, step=current_step)
                        print("Epoch {} training loss: {}".format(
                            current_step/steps_per_epoch, loss_for_logging), end="\r")

                    # ========== save checkpoints ==========
                    if args.checkpoint_path and current_step % 200 == 0:
                        self.save_checkpoint(args.checkpoint_path, epoch, current_step, model, self.optim, self.scheduler)
                    
                    current_step += 1
                
                print("\nEpoch {} avg training loss: {}".format(
                    epoch, sum(losses)/num_datapoints))
                '''

                # ========== validation ==========
                val_losses = []
                num_datapoints = 0
                preds = []
                labels = []
                with torch.no_grad():

                    for step, (batch, tokenized_inp) in enumerate(zip(val_dl, tokenized_val_helper)):
                        # ========== forward pass ==========
                        batch = {i:j.to(self.device) for i,j in batch.items()}
                        outputs = model(**batch)
                        loss = self.task.loss_function(outputs, batch)

                        # ========== compute metric ==========

                        preds.extend(
                            self.task.extract_answer_from_output(tokenized_inp, outputs)
                        )
                        import ipdb; ipdb.set_trace()
                        labels.extend(
                            self.task.extract_label_from_input(batch)
                        )

                        # SANDBOX
                        val_result = self.task.compute_metric(preds, labels)
                        import ipdb; ipdb.set_trace()

                        # ========== logging ==========
                        val_loss_for_logging = loss.detach().tolist()
                        val_losses.append(val_loss_for_logging*len(batch))
                        num_datapoints += len(batch)
                        print("Epoch {} validation loss: {}".format(
                            step/val_steps_per_epoch, val_loss_for_logging), end="\r")

                    self.wandb.log({"val/loss": sum(val_losses)/num_datapoints})
                    print("Epoch {} avg validation loss: {}".format(
                            epoch, sum(val_losses)/num_datapoints))
                    val_result = self.task.compute_metric(preds, labels)
                    print("Epoch {} validation acc: {}".format(
                        epoch, val_result))
                    self.wandb.log({"val/{}".format(i):j for i,j in val_result.items()})
                
                # ========== save checkpoints ==========
                if args.checkpoint_path:
                    self.save_checkpoint(args.checkpoint_path, epoch, len(train_dl), model, self.optim, self.scheduler)
       
            current_step = 0
        
        # ========== save checkpoints ==========
        except KeyboardInterrupt:
            print("Interrupted. Saving checkpoint...")
            self.save_checkpoint(args.checkpoint_path, epoch, current_step, model, self.optim, self.scheduler)
            raise

    # def evaluate(self, dl):
    #     pred_list = []
    #     label_list = []
    #     with torch.no_grad():
    #         for inp in dl:
    #             preds, labels = self.task.evaluate(inp, inp['label'])
    #             pred_list.extend(preds)
    #             label_list.extend(labels)
    
    #     result = self.task.metric.compute(
    #         predictions=pred_list,
    #         references=label_list,
    #     )

    #     return result

    def inference(self, dl):
        self.task.model.eval()
        infer_list = []
        with torch.no_grad():
            for inp in dl:
                preds = self.task.inference(inp)
                infer_list.extend(preds)
        return infer_list

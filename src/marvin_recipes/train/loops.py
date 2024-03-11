from torch import optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from marvin_recipes.misc.observability import TrainingTracker
from marvin_recipes.models.checkpoints import save_model_checkpoint, save_partial_training_checkpoint, \
    save_dataloader_state, load_partial_training_checkpoint, load_dataloader_state
from marvin_recipes.models.model_definitions import LanguageModel
import torch.distributed as dist
import torch
from marvin_recipes.train.utils import AnyPrecisionAdamW, TrainConfig


def get_optimizer(l_model: LanguageModel, config: TrainConfig):
    if config.optimizer.lower() == "adamw":
        optimizer = optim.AdamW(l_model.model.parameters(), lr=config.lr,
                                weight_decay=config.weight_decay)
    elif config.optimizer.lower() == "adam":
        optimizer = optim.Adam(l_model.model.parameters(), lr=config.lr,
                               weight_decay=config.weight_decay)
    elif config.optimizer.lower() == "anyadamw":
        optimizer = AnyPrecisionAdamW(l_model.model.parameters(), lr=config.lr,
                                      weight_decay=config.weight_decay)
    else:
        raise NotImplementedError(f"Optimizer {config.optimizer} not implemented")
    return optimizer


def get_scheduler(optimizer, config: TrainConfig):
    if config.scheduler.lower() == "steplr":
        scheduler = StepLR(optimizer, step_size=1, gamma=config.step_lr_gamma)
        print(f"##PARM: Gamma: {config.step_lr_gamma}")
    else:
        raise NotImplementedError(f"Scheduler {config.scheduler} not implemented")
    return scheduler


class SimpleFSDPTrainLoop:

    def __init__(self, l_model: LanguageModel, config: TrainConfig, local_rank: int, rank: int,
                 world_size: int, tracker: TrainingTracker):
        self.l_model = l_model
        self.config = config
        self.local_rank = local_rank
        self.rank = rank
        self.world_size = world_size
        self.model = self.l_model.model
        self.lr = self.config.lr
        self.data_module = self.config.data_module
        self.tokenizer = self.data_module.tokenizer
        self.gradient_acc_steps = self.config.gradient_accumulation_steps
        self.warmup_steps = self.config.warmup_steps
        self.train_dataloader = self.data_module.train_dataloader()
        self.num_mini_batches = len(self.train_dataloader)
        self.train_perp = []
        self.train_loss = []
        self.tracker = tracker
        self.checkpoint_steps = self.config.checkpoint_steps

    def step(self, batch, step, optimizer, global_step):
        batch = {key: batch[key].to(self.local_rank) for key in batch.keys()}
        loss = self.model(**batch, use_cache=False).loss
        # loss = self.model(**batch).loss
        loss = loss / self.gradient_acc_steps

        # regular backpropagation
        loss.backward()
        if (step + 1) % self.gradient_acc_steps == 0 or (step + 1) == self.num_mini_batches:
            optimizer_steps = (global_step + 1) // self.gradient_acc_steps
            if optimizer_steps <= self.warmup_steps:
                lr = self.lr * optimizer_steps / self.warmup_steps
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr
                if self.local_rank == 0:
                    tqdm.write(f"Warmup Step {optimizer_steps+1}/{self.warmup_steps}, setting lr to {lr}")

            optimizer.step()
            optimizer.zero_grad()

        return loss.detach().float()

    def log_step(self, step, pbar, epoch, loss, input_length, global_step=None):
        pbar.update(1)
        pbar.set_description(
            f"Training Epoch: {epoch + 1}/{self.config.max_epochs}, step {step}/{self.num_mini_batches} (loss: {loss})")
        self.tracker.log_metrics({"loss": loss.item(), "input_length": input_length}, step=global_step)

    def log_end_of_epoch(self, global_step=None):
        train_epoch_loss = self.train_loss[-1]
        train_perplexity = self.train_perp[-1]
        if self.local_rank == 0:
            tqdm.write(f"--> Epoch Training Loss: {train_epoch_loss}")
            tqdm.write(f"--> Epoch Training Perplexity: {train_perplexity}")
            tqdm.write(f"--> Synchronizing Processes...")
        self.tracker.log_metrics({"epoch_loss": train_epoch_loss, "epoch_perplexity": train_perplexity}, step=global_step)

    def init_train_metrics(self):
        self.train_perp = []
        self.train_loss = []

    def calculate_epoch_metrics(self, total_loss):
        train_epoch_loss = (total_loss / self.num_mini_batches) / self.world_size
        train_perplexity = torch.exp(train_epoch_loss)
        self.train_perp.append(train_perplexity)
        self.train_loss.append(train_epoch_loss)
        return train_epoch_loss, train_perplexity

    def log_data_sample(self, batch, tokenizer, local_rank, step, interval=40, remote_interval=10):
        if local_rank == 0 and (step % interval == 0 or step % remote_interval == 0):
            input_ids = batch["input_ids"]
            labels = batch["labels"].clone()
            input_decoded = tokenizer.decode(input_ids[0], skip_special_tokens=False)
            try:
                labels[labels == -100] = tokenizer.bos_token_id
                labels_decoded = tokenizer.decode(labels[0], skip_special_tokens=False)
            except:
                labels_decoded = "None"

            if step % interval == 0:
                tqdm.write(f"Input: {input_decoded}")
                tqdm.write(f"Labels: {labels_decoded}")
                tqdm.write(f"Input Shape: {input_ids.shape}")

            if step % remote_interval == 0:
                self.tracker.trace_input(input_decoded, labels_decoded, f"{input_ids.shape}")

    def start(self):
        self.init_train_metrics()
        optimizer = get_optimizer(self.l_model, self.config)
        resumed, resume_data = self.try_to_resume_training(optimizer)
        scheduler = get_scheduler(optimizer, self.config)
        if resumed:
            global_step = resume_data["step"]
            scheduler.load_state_dict(resume_data["scheduler_state"])
            current_epoch_loss = resume_data["epoch_loss"].to(self.local_rank)
            current_epoch = resume_data["epoch"]
            step_in_epoch = resume_data["step_in_epoch"]
            current_samples_in_epoch = step_in_epoch * self.config.mini_batch_size
            self.train_perp = resume_data["train_perp"]
            self.train_loss = resume_data["train_loss"]
        else:
            global_step = 0
            current_epoch_loss = 0.0
            current_epoch = 0
            current_samples_in_epoch = 0
            step_in_epoch = 0

        max_epochs = self.config.max_epochs if self.config.max_epochs else 1
        for epoch in range(current_epoch, max_epochs):
            self.model.train()
            num_samples_in_epoch = current_samples_in_epoch
            current_samples_in_epoch = 0
            total_loss = current_epoch_loss
            current_epoch_loss = 0.0

            pbar = tqdm(colour="blue", desc=f"Training Epoch: {epoch + 1}", total=self.num_mini_batches,
                        dynamic_ncols=True, initial=step_in_epoch)

            step = step_in_epoch
            step_in_epoch = 0
            for batch in self.train_dataloader:
                global_step += 1
                step += 1
                mini_batch_size = len(batch['input_ids'])
                num_samples_in_epoch += mini_batch_size
                self.log_data_sample(batch, self.tokenizer, self.local_rank, step)
                step_loss = self.step(batch, step, optimizer, global_step)
                step_loss = step_loss / (8/self.gradient_acc_steps) # TODO: Used to scale the reported loss to be equivalent to previous runs (Remove and fix for future releases)
                self.log_step(step, pbar, epoch, step_loss, batch["input_ids"].shape[-1], global_step=global_step)
                total_loss += step_loss
                if self.checkpoint_steps and global_step % self.checkpoint_steps == 0:
                    save_partial_training_checkpoint(self.model, optimizer, self.config)
                    scheduler.state_dict()
                    save_dataloader_state(num_samples_in_epoch, epoch, mini_batch_size, global_step, self.config,
                                          total_loss, self.train_perp, self.train_loss, scheduler)
                    if self.local_rank == 0:
                        tqdm.write(f"Checkpointing at step {global_step} and epoch {epoch} Done!")

            # End of epoch
            scheduler.step()

            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
            self.calculate_epoch_metrics(total_loss)
            self.log_end_of_epoch(global_step=global_step)
            save_model_checkpoint(self.model, self.rank, epoch=epoch, save_dir=self.config.save_dir,
                                  chk_name=self.config.run_name)
            pbar.close()
            dist.barrier()
        self.tracker.close()
        if self.local_rank == 0:
            print("Training Done!")

    def try_to_resume_training(self, optimizer):
        if self.local_rank == 0:
            print("Trying to resume training...")
        resumed = load_partial_training_checkpoint(
            self.model, optimizer, self.local_rank, self.config)

        if resumed:
            state_dict = load_dataloader_state(self.data_module, self.config)
            if self.local_rank == 0:
                print("Resuming Training!")

        else:
            state_dict = None
            if self.local_rank == 0:
                print("Starting Training from scratch!")
        return resumed, state_dict

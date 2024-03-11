import shutil
from pathlib import Path
import torch
from torch.distributed.fsdp import StateDictType, FullStateDictConfig, LocalStateDictConfig, ShardedStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed._shard.checkpoint import FileSystemReader, FileSystemWriter, save_state_dict, load_state_dict, \
    load_sharded_optimizer_state_dict
import torch.distributed as dist
import logging
import time

from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer

from marvin_recipes.data.modules import DataModule
from marvin_recipes.distributed.utils import get_process_info
from marvin_recipes.train.utils import TrainConfig

logger = logging.getLogger(__name__)


def save_model_checkpoint(model, rank, save_dir, chk_name, epoch=1):
    dist.barrier()
    f_type = StateDictType.FULL_STATE_DICT
    cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

    with FSDP.state_dict_type(model, f_type, cfg):
        cpu_state = model.state_dict()
        logger.info(f"saving process: rank {rank} done with model state_dict\n")

    if rank == 0:
        print(f"--> saving model ...")
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_full_path = str(save_dir) + f"/{chk_name}_{epoch}.pt"
        torch.save(cpu_state, save_full_path)
        logger.info(f"model checkpoint saved for epoch {epoch} at {save_full_path}\n")
    else:
        logger.info(f"--> skipping model saving for rank {rank}\n")


def organize_partial_directory(save_dir, job_name, rank):
    if rank != 0:
        return

    current_dir = Path(save_dir) / job_name / "latest"
    if current_dir.exists():
        logger.info(f"--> Moving existing partial checkpoint directory {current_dir}")
        new_dir = Path(save_dir) / job_name / f"prev"
        # Delete existing prev directory
        if new_dir.exists():
            shutil.rmtree(new_dir)
        # Move existing latest directory to prev
        shutil.move(current_dir, new_dir)
    else:
        logger.info(f"--> no existing partial checkpoint directory {current_dir}")
    current_dir.mkdir(parents=True, exist_ok=True)


def save_partial_training_checkpoint(model, optimizer, config: TrainConfig):
    save_dir = config.job_checkpoint_dir
    local_rank, rank, world_size = get_process_info()
    organize_partial_directory(save_dir, config.run_name, rank)
    dist.barrier()

    save_dir = Path(save_dir) / config.run_name / "latest" / "model_and_optim"
    if local_rank == 0:
        save_dir.mkdir(parents=True, exist_ok=True)
    writer = FileSystemWriter(save_dir)
    t0 = time.perf_counter()
    cfg = ShardedStateDictConfig(offload_to_cpu=False)
    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT, cfg):
        state_dict = {
            "model": model.state_dict(),
            "optimizer": FSDP.optim_state_dict(model, optimizer),
        }
        save_state_dict(state_dict, writer)

    t1 = time.perf_counter()

    if local_rank == 0:
        logger.info(f"--> Saved model partial checkpoint in {t1 - t0:.2f} seconds\n")


def save_dataloader_state(num_samples_in_epoch, epoch, mini_batch_size, global_step, config: TrainConfig, epoch_loss,
                          train_perp, train_loss, scheduler: LRScheduler):
    local_rank, rank, world_size = get_process_info()
    data_module = config.data_module
    save_dir = config.job_checkpoint_dir
    run_name = config.run_name
    if rank == 0:
        dataset_state = data_module.state_dict(from_beginning=False, num_samples=num_samples_in_epoch*world_size)
        state_dict = {
            "step": global_step,
            "epoch_loss": epoch_loss,
            "scheduler_state": scheduler.state_dict(),
            "epoch": epoch,
            "train_perp": train_perp,
            "train_loss": train_loss,
            "step_in_epoch": num_samples_in_epoch // mini_batch_size,
            "dataset_state": dataset_state
        }
        state_list = [state_dict]
    else:
        state_list = [None]

    dist.broadcast_object_list(state_list, src=0)
    if local_rank == 0:
        save_file = Path(save_dir) / run_name / "latest" / "data_state_dict.pt"
        state_dict = state_list[0]
        torch.save(state_dict, save_file)
        logger.info(f"Saved dataloader state dict at {save_file}\n")


def load_dataloader_state(data_module: DataModule, config: TrainConfig):
    local_rank, rank, world_size = get_process_info()

    save_file = Path(config.job_checkpoint_dir) / config.run_name / "latest" / "data_state_dict.pt"
    state_dict = torch.load(save_file)
    data_module.load_state_dict(state_dict.pop("dataset_state"))
    if local_rank == 0:
        logger.info(f"Loaded dataloader state dict from {save_file}\n")

    return state_dict


def load_partial_training_checkpoint(model: torch.nn.Module, optimizer: Optimizer, l_rank, config: TrainConfig):
    save_dir = Path(config.job_checkpoint_dir) / config.run_name / "latest" / "model_and_optim"

    if not save_dir.exists():
        if l_rank == 0:
            logger.info(f"--> No model checkpoint found at {save_dir}\n")
        return False

    t0 = time.perf_counter()

    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
        state_dict = {"model": model.state_dict()}

        load_state_dict(state_dict=state_dict, storage_reader=FileSystemReader(save_dir))
        model.load_state_dict(state_dict["model"])

        optim_state = load_sharded_optimizer_state_dict(
            model_state_dict=state_dict["model"],
            optimizer_key="optimizer",
            storage_reader=FileSystemReader(save_dir),
        )

        flattened_osd = FSDP.optim_state_dict_to_load(
            model, optimizer, optim_state["optimizer"]
        )
        optimizer.load_state_dict(flattened_osd)

    dist.barrier()
    t1 = time.perf_counter()

    if l_rank == 0:
        logger.info(f"--> Loaded model partial checkpoint in {t1 - t0:.2f} seconds\n")

    return True

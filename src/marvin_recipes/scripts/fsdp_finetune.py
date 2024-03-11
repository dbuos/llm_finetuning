import logging

import fire
from marvin_recipes.data.modules import stream_data_module_instruct
from marvin_recipes.distributed.utils import setup_distributed_process
from marvin_recipes.distributed.model_preparation import shard_and_distribute
from marvin_recipes.misc.observability import get_tracker
from marvin_recipes.misc.utils import print_model_size, log_train_start
from marvin_recipes.models.load import load_from_pretrained_chk
from marvin_recipes.train.loops import SimpleFSDPTrainLoop, TrainConfig

logging.basicConfig(level=logging.INFO)


def launch(
        model_chk: str,
        seed: int = 42,
        dist_backend: str = "nccl",
        act_checkpointing: bool = True,
        data_blend: str = None,
        remote_dataset: str = "s3://some-bucket/early_demo/qa_marvin_v01_fixed",
        save_dir: str = "model_checkpoints",
        job_checkpoint_dir: str = None,
        checkpoint_steps: int = None,
        run_name: str = "fine-tuned",
        optimizer: str = "adamw",
        with_splits: bool = False,
        grad_acc_steps: int = 4,
        num_epochs: int = 3,
        data_loader_workers: int = 8,
        external_logger: str = "wandb",
        log_project_name: str = "marvin_fine_tuning",
        warmup_steps: int = 10,
        local_data_path: str = None,
        lr: float = 1e-4,
        language_modeling: bool = False,
        mini_batch_size: int = 4):
    local_rank, rank, world_size = setup_distributed_process(seed, dist_backend)
    tracker = get_tracker(external_logger, log_project_name, run_name, rank=rank, local_rank=0)  # Log in all ranks
    h_params = {
        "lr": lr,
        "mini_batch_size": mini_batch_size,
        "grad_acc_steps": grad_acc_steps,
        "num_epochs": num_epochs,
        "model_chk": model_chk,
        "seed": seed,
        "dist_backend": dist_backend,
        "activation_checkpointing": act_checkpointing,
        "warmup_steps": warmup_steps,
        "optimizer": optimizer,
        "world_size": world_size,
        "data_loader_workers": data_loader_workers,
        "language_modeling": language_modeling
    }
    if data_blend:
        h_params["data_blend"] = data_blend
    else:
        h_params["remote_dataset"] = remote_dataset
    tracker.add_hparams(h_params)

    l_model = load_from_pretrained_chk(model_chk)
    print_model_size(l_model.model, local_rank, run_name)

    l_model = shard_and_distribute(l_model, act_checkpointing, local_rank)
    data_module = stream_data_module_instruct(
        model_chk,
        remote=remote_dataset,
        data_blend_path=data_blend,
        with_splits=with_splits,
        num_workers=data_loader_workers,
        local_dir=local_data_path,
        mini_batch_size=mini_batch_size,
        lm=language_modeling
    )

    log_train_start(local_rank, data_module, l_model)

    config = TrainConfig(
        max_epochs=num_epochs,
        lr=lr,
        save_dir=save_dir,
        run_name=run_name,
        data_module=data_module,
        gradient_accumulation_steps=grad_acc_steps,
        warmup_steps=warmup_steps,
        optimizer=optimizer,
        job_checkpoint_dir=job_checkpoint_dir,
        mini_batch_size=mini_batch_size,
        checkpoint_steps=checkpoint_steps
    )

    loop = SimpleFSDPTrainLoop(l_model, config, local_rank, rank, world_size, tracker)
    loop.start()


if __name__ == "__main__":
    fire.Fire(launch)

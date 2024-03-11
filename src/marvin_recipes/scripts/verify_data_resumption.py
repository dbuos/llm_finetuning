import warnings

warnings.filterwarnings("ignore")
from transformers import AutoTokenizer
from marvin_recipes.data.modules import stream_data_module_instruct
from marvin_recipes.distributed.utils import get_process_info
from torch import distributed as dist
from fire import Fire
from tqdm import tqdm
from hashlib import sha256
from pathlib import Path
import pickle
import torch

from marvin_recipes.misc.utils import set_seed


def input_ids_to_str(input_ids, tokenizer):
    input_str = ""
    for i in range(len(input_ids)):
        input_str += tokenizer.decode(input_ids[i])
    input_str = sha256(input_str.encode()).hexdigest()
    return f"{input_str}-{input_ids.shape[-1]}"


def main(
        tmp_dir: str,
        model_chk: str,
        data_blend_path: str,
        mini_batch_size: int = 2,
        epochs: int = 1,
        result_dir: str = "data_resumption",
        recover_from: str = None,
        checkpoint_steps: int = None,
        checkpoint_dir: str = None,
        seed: int = 42,
        dist_backend: str = "gloo"):
    set_seed(seed)
    dist.init_process_group(dist_backend)
    local_rank, rank, world_size = get_process_info()
    tokenizer = AutoTokenizer.from_pretrained(model_chk)

    if checkpoint_steps:
        assert checkpoint_dir is not None, "Must provide checkpoint_dir if checkpoint_steps is not None"

    data_module = stream_data_module_instruct(
        model_chk,
        data_blend_path=data_blend_path,
        num_workers=1,
        local_dir=tmp_dir,
        mini_batch_size=mini_batch_size)

    shard_batch_list = []
    data_loader = data_module.train_dataloader()
    step = 0
    current_epoch = 0
    total = len(data_loader)
    initial = 0
    current_num_samples = 0

    if recover_from is not None:
        recover_dir = Path(recover_from)
        assert recover_dir.exists(), f"Recover dir {recover_dir} does not exist"
        data_state_dict = torch.load(recover_dir / f"data_state_dict.pt")
        current_epoch = data_state_dict["epoch"]
        current_num_samples = data_state_dict["step_in_epoch"] * mini_batch_size
        step = data_state_dict["step"]
        initial = data_state_dict["step_in_epoch"]
        dataset_state = data_state_dict["dataset_state"]
        data_loader.dataset.load_state_dict(dataset_state)
        resume_file = recover_dir / f"resume_{local_rank}.pt"
        with open(resume_file, 'rb') as f:
            shard_batch_list = pickle.load(f)
        if local_rank == 0:
            print(f"Resuming from step {step} and epoch {current_epoch}")

    for epoch in range(current_epoch, epochs):
        num_samples_so_far = current_num_samples
        current_num_samples = 0
        pbar = tqdm(total=total, initial=initial) if local_rank == 0 else None
        for batch in data_loader:
            step += 1
            num_samples_so_far += len(batch['input_ids'])
            input_ids = batch['input_ids']
            input_str = input_ids_to_str(input_ids, tokenizer)
            shard_batch_list.append(input_str)

            dist.barrier()
            # Save state dict
            if checkpoint_steps and step % checkpoint_steps == 0:
                c_dir = Path(checkpoint_dir) / f"step_{step}"
                c_dir.mkdir(exist_ok=True, parents=True)
                resume_file = c_dir / f"resume_{local_rank}.pt"
                with open(resume_file, 'wb') as f:
                    pickle.dump(shard_batch_list, f)
                if local_rank == 0:
                    tqdm.write(f"Saving checkpoint at step {step} and epoch {epoch}")
                    checkpoint_file = c_dir / f"data_state_dict.pt"
                    state = data_module.train_dataset.state_dict(from_beginning=False,
                                                                 num_samples=num_samples_so_far * world_size)
                    state_dict = {
                        "step": step,
                        "epoch": epoch,
                        "step_in_epoch": num_samples_so_far // mini_batch_size,
                        "dataset_state": state
                    }
                    torch.save(state_dict, checkpoint_file)
                    # with open(checkpoint_file, 'wb') as f:
                    #     pickle.dump(state_dict, f)
                dist.barrier()
            if local_rank == 0:
                pbar.update(1)
        initial = 0
        if local_rank == 0:
            pbar.close()
    dist.barrier()
    # shard_batch_set = set(shard_batch_list)
    # print(f"Rank {local_rank} found {len(shard_batch_list)} batches")

    gather_list = [[] for _ in range(world_size)]
    dist.all_gather_object(gather_list, shard_batch_list)

    if local_rank == 0:
        all_batches = []
        for elem in gather_list:
            all_batches.extend(elem)
        all_batches_set = set(all_batches)
        print(f"\n\nSize of all batches set: {len(all_batches_set)}")
        print(f"Size of all batches list: {len(all_batches)}")
        num_duplicates = len(all_batches) - len(all_batches_set)
        print(f"Found {num_duplicates} duplicate batches")

    result_dir = Path(result_dir)
    result_dir.mkdir(exist_ok=True)
    result_file = result_dir / f"result_{local_rank}.pkl"

    if not result_file.exists():
        with open(result_file, 'wb') as f:
            pickle.dump(shard_batch_list, f)
    else:
        with open(result_file, 'rb') as f:
            prev_result = pickle.load(f)

        pass_test = True
        for i, elem in enumerate(shard_batch_list):
            if elem != prev_result[i]:
                pass_test = False
                break

        gather_list = [[] for _ in range(world_size)]
        dist.all_gather_object(gather_list, (local_rank, pass_test))

        if local_rank == 0:
            all_pass = all(result for _rank, result in gather_list)
            if all_pass:
                print("All ranks passed the test")
            else:
                print("Some ranks failed the test")
                print(gather_list)


if __name__ == '__main__':
    Fire(main)

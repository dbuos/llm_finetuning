import os
import random

import fire
import torch
from streaming import StreamingDataset, Stream
from torch.distributed import init_process_group
import torch.distributed as dist
from torch.utils.data import DataLoader
import time
import json

from marvin_recipes.data.modules import stream_data_module_instruct

ds_to_id = {
    "raw_datasets/alpaca_4_only_qa": 0,
    "raw_datasets/lima_only_qa/qa_lima.json": 1,
    "raw_datasets/oass_en_only_qa": 2,
    "raw_datasets/dolly_es_only_qa": 3,
    "raw_datasets/synthetic_marvin_01/qa_marvin.json": 4,
    "raw_datasets/oass_es_only_qa": 5
}

id_to_ds = dict([(v, k) for k, v in ds_to_id.items()])


# def create_streams(base_url, temp_base, with_proportions=False):
#     if not with_proportions:
#         streams = [
#             Stream(remote=f'{base_url}/alpaca_4_only_qa_short_02', choose=400, local=f'{temp_base}/t1'),
#             Stream(remote=f'{base_url}/lima_only_qa_short_02', choose=800, local=f'{temp_base}/t2'),
#             Stream(remote=f'{base_url}/dolly_es_only_qa_short_02', choose=700, local=f'{temp_base}/t3'),
#             Stream(remote=f'{base_url}/oass_es_only_qa_short_02', choose=1000, local=f'{temp_base}/t4'),
#             Stream(remote=f'{base_url}/oass_en_only_qa_short_02', choose=500, local=f'{temp_base}/t5'),
#             Stream(remote=f'{base_url}/qa_marvin_short_02', choose=5000, local=f'{temp_base}/t6'),
#         ]
#     else:
#         streams = [
#             Stream(remote=f'{base_url}/qa_marvin_short_02', proportion=0.5, local=f'{temp_base}/t6'),
#             Stream(remote=f'{base_url}/alpaca_4_only_qa_short_02', proportion=0.1, local=f'{temp_base}/t1'),
#             Stream(remote=f'{base_url}/lima_only_qa_short_02', proportion=0.1, local=f'{temp_base}/t2'),
#             Stream(remote=f'{base_url}/dolly_es_only_qa_short_02', proportion=0.1, local=f'{temp_base}/t3'),
#             Stream(remote=f'{base_url}/oass_es_only_qa_short_02', proportion=0.1, local=f'{temp_base}/t4'),
#             Stream(remote=f'{base_url}/oass_en_only_qa_short_02', proportion=0.1, local=f'{temp_base}/t5'),
#         ]
#     return streams


def print_process_info():
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    master_addr = os.environ['MASTER_ADDR']
    master_port = os.environ['MASTER_PORT']
    print(
        f'rank: {rank}, world_size: {world_size}, local_rank: {local_rank}, master_addr: {master_addr}, master_port: {master_port}')
    pass


def count_elems_in_epoch(d_loader):
    count = 0
    sources = {}
    uuids = []
    for i, batch in enumerate(d_loader):
        prompts = batch['prompt']
        uuids.extend(batch['uuid'])
        for s in batch['source_dataset']:
            if s not in sources:
                sources[s] = 0
            sources[s] += 1
        count += len(prompts)
    return count, sources, uuids


def main(temp_base="temp_010",
         workers: int = 1,
         data_blend_path: str = None):
    init_process_group(backend='gloo')
    print_process_info()

    # streams = create_streams(base_url, temp_base, with_proportions=with_proportions)
    data_module = stream_data_module_instruct(
        "meta-llama/Llama-2-7b-hf",
        data_blend_path=data_blend_path,
        with_splits=False,
        num_workers=workers,
        local_dir=temp_base,
        raw=True,
        mini_batch_size=8)
    dl = data_module.train_dataloader()
    # dl = DataLoader(dataset, batch_size=8, drop_last=True, num_workers=workers)

    count, sources_dict, uuids = count_elems_in_epoch(dl)
    _, _, uuids_e2 = count_elems_in_epoch(dl)

    sources_pretty = json.dumps(sources_dict, indent=4)

    local_rank = int(os.environ['LOCAL_RANK'])
    if local_rank == 0:
        print(f"Using {workers} workers")

    report_str = f"""#### Report for local rank {local_rank} ####
Total number of elements in epoch: {count}
Sources: 
{sources_pretty}
#### End of report for local rank {local_rank} ####
    """

    # Random sleep between 200 and 1500 ms
    sleep_time = 0.2 + (1.3 - 0.2) * random.random()
    time.sleep(sleep_time)
    print(report_str)

    dist.barrier()
    world_size = int(os.environ['WORLD_SIZE'])

    elem_count = torch.zeros((len(id_to_ds),), dtype=torch.int64)
    for k, v in sources_dict.items():
        elem_count[ds_to_id[k]] = v

    dist.all_reduce(elem_count, op=dist.ReduceOp.SUM)

    # Total Sources count
    total_sources_counts = {}
    for k, v in ds_to_id.items():
        total_sources_counts[k] = elem_count[v].item()

    total_sources_proportions = {}
    total_elems = elem_count.sum().item()
    for k, v in ds_to_id.items():
        total_sources_proportions[k] = elem_count[v].item() / total_elems

    total_sources_counts_pretty = json.dumps(total_sources_counts, indent=4)
    total_sources_proportions_pretty = json.dumps(total_sources_proportions, indent=4)

    # UUIDs epoch 1
    gather_list = [[] for _ in range(world_size)]
    dist.all_gather_object(gather_list, uuids)
    all_uuids = [item for sublist in gather_list for item in sublist]

    # UUIDs epoch 2
    gather_list_e2 = [[] for _ in range(world_size)]
    dist.all_gather_object(gather_list_e2, uuids_e2)
    all_uuids_e2 = [item for sublist in gather_list_e2 for item in sublist]

    if local_rank == 0:
        non_repeated_uuids = set(all_uuids)
        non_repeated_uuids_e2 = set(all_uuids_e2)
        num_common_uuids = len(non_repeated_uuids.intersection(non_repeated_uuids_e2))
        print()
        print("All reports gathered, printing final report")
        print()
        final_report = f"""#### Final report ####
Total number of elements in epoch: {elem_count.sum().item()}
UUIDs: {len(all_uuids)}
UUIDs without repetitions: {len(non_repeated_uuids)}
Num of repeated UUIDs: {len(all_uuids) - len(non_repeated_uuids)}
Sources counts:
{total_sources_counts_pretty}
Sources proportions:
{total_sources_proportions_pretty}
Num of common UUIDs between epoch 1 and epoch 2: {num_common_uuids}
"""
        print(final_report)


if __name__ == '__main__':
    fire.Fire(main)

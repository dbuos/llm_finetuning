from typing import Any
from streaming import StreamingDataset


class MarvinInstructStreamDataset(StreamingDataset):
    def __init__(self, batch_size, tokenizer, raw=False, **kwargs):
        super().__init__(batch_size=batch_size, predownload=batch_size * 12, shuffle_seed=31,
                         num_canonical_nodes=64, shuffle_block_size=262144, **kwargs)
        self.tokenizer = tokenizer
        self.raw = raw

    def __getitem__(self, idx: int) -> Any:
        obj = super().__getitem__(idx)
        if self.raw:
            return obj

        bos = self.tokenizer.bos_token if self.tokenizer.bos_token else ""
        prompt = self.tokenizer.encode(bos + obj["prompt"], add_special_tokens=False)
        eos = self.tokenizer.eos_token if self.tokenizer.eos_token else ""
        answer = self.tokenizer.encode(obj["model_answer"] + eos, add_special_tokens=False)
        return {
            "input_ids": prompt + answer,
            "labels": [-100] * len(prompt) + answer,
        }


class MarvinLanguageModelStreamDataset(StreamingDataset):
    def __init__(self, batch_size, tokenizer, raw=False, **kwargs):
        super().__init__(batch_size=batch_size, predownload=batch_size * 12, shuffle_seed=31,
                         num_canonical_nodes=64, shuffle_block_size=262144, **kwargs)
        self.tokenizer = tokenizer
        self.raw = raw

    def __getitem__(self, idx: int) -> Any:
        obj = super().__getitem__(idx)
        if self.raw:
            return obj
        bos = self.tokenizer.bos_token if self.tokenizer.bos_token else ""
        eos = self.tokenizer.eos_token if self.tokenizer.eos_token else ""
        prompt = self.tokenizer.encode(bos + obj["prompt"] + eos, add_special_tokens=False)
        # answer = self.tokenizer.encode(obj["model_answer"] + self.tokenizer.eos_token, add_special_tokens=False)
        return {
            "input_ids": prompt,
            "labels": prompt,
        }

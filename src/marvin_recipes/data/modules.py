from dataclasses import dataclass
from pathlib import Path

from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase, DataCollatorForSeq2Seq, AutoTokenizer

from marvin_recipes.data.stream import MarvinInstructStreamDataset, MarvinLanguageModelStreamDataset
from streaming import Stream
import json


@dataclass
class DataModule:
    tokenizer: PreTrainedTokenizerBase

    def train_dataloader(self) -> DataLoader:
        raise NotImplementedError

    def test_dataloader(self) -> DataLoader:
        raise NotImplementedError

    def load_state_dict(self, state_dict):
        raise NotImplementedError

    def state_dict(self, **kwargs):
        raise NotImplementedError


class InstructStreamingDataModule(DataModule):

    def __init__(self, tokenizer: PreTrainedTokenizerBase,
                 mini_batch_size: int,
                 with_splits: bool = True,
                 num_workers: int = 1,
                 shuffle: bool = True,
                 raw: bool = False,
                 **kwargs):
        super().__init__(tokenizer)

        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

        collator = DataCollatorForSeq2Seq(tokenizer=tokenizer) if not raw else None

        train_split = None
        if with_splits:
            train_split = "train"

        self.train_dataset = MarvinInstructStreamDataset(
            batch_size=mini_batch_size,
            tokenizer=tokenizer,
            raw=raw,
            shuffle=shuffle,
            split=train_split, **kwargs)

        self.train_dl = DataLoader(
            self.train_dataset,
            batch_size=mini_batch_size,
            drop_last=True,
            pin_memory=True,
            num_workers=num_workers,
            collate_fn=collator)

        if with_splits:
            self.test_dataset = MarvinInstructStreamDataset(
                batch_size=mini_batch_size,
                tokenizer=tokenizer,
                raw=raw,
                shuffle=False,
                split="test", **kwargs)

            self.test_dl = DataLoader(
                self.test_dataset,
                batch_size=mini_batch_size,
                drop_last=True,
                pin_memory=True,
                num_workers=num_workers,
                collate_fn=collator)
        else:
            self.test_dataset = None
            self.test_dl = None

    def train_dataloader(self):
        return self.train_dl

    def test_dataloader(self):
        return self.test_dl

    def load_state_dict(self, state_dict):
        self.train_dataset.load_state_dict(state_dict)

    def state_dict(self, **kwargs):
        return self.train_dataset.state_dict(**kwargs)


class LMStreamingDataModule(DataModule):

    def __init__(self, tokenizer: PreTrainedTokenizerBase,
                 mini_batch_size: int,
                 with_splits: bool = True,
                 num_workers: int = 1,
                 shuffle: bool = True,
                 raw: bool = False,
                 **kwargs):
        super().__init__(tokenizer)

        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

        collator = DataCollatorForSeq2Seq(tokenizer=tokenizer) if not raw else None

        train_split = None
        if with_splits:
            train_split = "train"

        self.train_dataset = MarvinLanguageModelStreamDataset(
            batch_size=mini_batch_size,
            tokenizer=tokenizer,
            raw=raw,
            shuffle=shuffle,
            split=train_split, **kwargs)

        self.train_dl = DataLoader(
            self.train_dataset,
            batch_size=mini_batch_size,
            drop_last=True,
            pin_memory=True,
            num_workers=num_workers,
            collate_fn=collator)

        if with_splits:
            self.test_dataset = MarvinLanguageModelStreamDataset(
                batch_size=mini_batch_size,
                tokenizer=tokenizer,
                raw=raw,
                shuffle=False,
                split="test", **kwargs)

            self.test_dl = DataLoader(
                self.test_dataset,
                batch_size=mini_batch_size,
                drop_last=True,
                pin_memory=True,
                num_workers=num_workers,
                collate_fn=collator)
        else:
            self.test_dataset = None
            self.test_dl = None

    def train_dataloader(self):
        return self.train_dl

    def test_dataloader(self):
        return self.test_dl

    def load_state_dict(self, state_dict):
        self.train_dataset.load_state_dict(state_dict)

    def state_dict(self, **kwargs):
        return self.train_dataset.state_dict(**kwargs)


def single_stream_data_module_instruct(tokenizer_chck: str,
                                       remote: str,
                                       mini_batch_size: int,
                                       with_splits: bool = False,
                                       num_workers: int = 8,
                                       raw: bool = False,
                                       shuffle: bool = True,
                                       local_dir: str = None) -> InstructStreamingDataModule:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_chck)
    return InstructStreamingDataModule(
        tokenizer=tokenizer,
        mini_batch_size=mini_batch_size,
        with_splits=with_splits,
        num_workers=num_workers,
        shuffle=shuffle,
        raw=raw,
        remote=remote,
        local=local_dir
    )


def blend_stream_data_module_instruct(tokenizer_chck: str,
                                      data_blend_path: str,
                                      mini_batch_size: int,
                                      with_splits: bool = False,
                                      num_workers: int = 8,
                                      raw: bool = False,
                                      shuffle: bool = True,
                                      local_dir: str = None) -> InstructStreamingDataModule:
    # read data_blend_path as json
    data_blend = json.load(open(data_blend_path, 'r'))
    streams_def = data_blend['streams']
    epoch_size = None
    if 'proportion' in streams_def[0]:
        epoch_size = data_blend['epoch_size']

    streams = []
    for i, stream_params in enumerate(streams_def):
        if local_dir:
            stream_params['local'] = Path(local_dir) / f"stream_{i}"
        stream = Stream(**stream_params)
        streams.append(stream)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_chck)
    return InstructStreamingDataModule(
        tokenizer=tokenizer,
        mini_batch_size=mini_batch_size,
        with_splits=with_splits,
        num_workers=num_workers,
        shuffle=shuffle,
        raw=raw,
        streams=streams,
        epoch_size=epoch_size
    )


def blend_stream_data_module_lm(tokenizer_chck: str,
                                data_blend_path: str,
                                mini_batch_size: int,
                                with_splits: bool = False,
                                num_workers: int = 8,
                                raw: bool = False,
                                shuffle: bool = True,
                                local_dir: str = None) -> LMStreamingDataModule:
    # read data_blend_path as json
    data_blend = json.load(open(data_blend_path, 'r'))
    streams_def = data_blend['streams']
    epoch_size = None
    if 'proportion' in streams_def[0]:
        epoch_size = data_blend['epoch_size']

    streams = []
    for i, stream_params in enumerate(streams_def):
        if local_dir:
            stream_params['local'] = Path(local_dir) / f"stream_{i}"
        stream = Stream(**stream_params)
        streams.append(stream)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_chck)
    return LMStreamingDataModule(
        tokenizer=tokenizer,
        mini_batch_size=mini_batch_size,
        with_splits=with_splits,
        num_workers=num_workers,
        shuffle=shuffle,
        raw=raw,
        streams=streams,
        epoch_size=epoch_size
    )


def stream_data_module_instruct(tokenizer_chck: str,
                                mini_batch_size: int,
                                data_blend_path: str = None,
                                remote: str = None,
                                with_splits: bool = False,
                                raw: bool = False,
                                num_workers: int = 8,
                                shuffle: bool = True,
                                local_dir: str = None,
                                lm: bool = False,
                                ) -> DataModule:
    if data_blend_path and not lm:
        return blend_stream_data_module_instruct(tokenizer_chck, data_blend_path, mini_batch_size,
                                                 with_splits, num_workers, raw, shuffle, local_dir)
    elif data_blend_path and lm:
        return blend_stream_data_module_lm(tokenizer_chck, data_blend_path, mini_batch_size,
                                           with_splits, num_workers, raw, shuffle, local_dir)
    else:
        return single_stream_data_module_instruct(tokenizer_chck, remote, mini_batch_size,
                                                  with_splits, num_workers, raw, shuffle, local_dir)

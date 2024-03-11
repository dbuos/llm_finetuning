import re

from datasets import load_dataset

exp_user_compiled = re.compile(r'<\|im_start\|>user\n(.*?)<\|im_end\|>', re.DOTALL)
exp_assistant_compiled = re.compile(r'<\|im_start\|>assistant\n(.*?)<\|im_end\|>', re.DOTALL)


def extract_turns(elem):
    user_turn = exp_user_compiled.findall(elem['text'])[0]
    assistant_turn = exp_assistant_compiled.findall(elem['text'])[0]
    return {"question": user_turn, "answer": assistant_turn}


def count_words(elem):
    return {'num_words': len(elem['question'].split())}


def process_and_save(dataset_name, save_path):
    oasst = load_dataset(dataset_name)
    # Only 2 turns
    oasst = oasst.filter(lambda x: x['num_turns'] == 2)
    oasst = oasst.map(extract_turns)
    oasst = oasst.map(count_words)
    oasst = oasst.filter(lambda x: x['num_words'] >= 3)
    # Only question and answer columns
    oasst = oasst.select_columns(['question', 'answer'])
    oasst['train'].save_to_disk(save_path)


if __name__ == '__main__':
    process_and_save(dataset_name="dbuos/oasst_top1_es", save_path="raw_datasets/oass_es_only_qa")

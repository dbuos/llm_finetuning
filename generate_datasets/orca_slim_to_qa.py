from datasets import load_dataset


def to_qa_columns(elem):
    assert elem['conversations'][0]['from'] == 'system'
    system_msg = elem['conversations'][0]['value']

    assert elem['conversations'][1]['from'] == 'human'
    question = elem['conversations'][1]['value']

    assert elem['conversations'][2]['from'] == 'gpt'
    answer = elem['conversations'][2]['value']

    qa_map = {
        'question': f"{system_msg}\n---\n{question}",
        'answer': answer
    }

    return qa_map


if __name__ == '__main__':
    orca = load_dataset("Open-Orca/SlimOrca-Dedup")
    orca = orca.map(to_qa_columns, remove_columns=['conversations'])
    orca['train'].save_to_disk('raw_datasets/orca_slim_simple_qa')

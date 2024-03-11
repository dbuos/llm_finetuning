from datasets import load_dataset

if __name__ == '__main__':
    alpaca_4 = load_dataset("vicgalle/alpaca-gpt4")
    alpaca_4 = alpaca_4['train']
    alpaca_4 = alpaca_4.rename_columns({'instruction': 'question', 'output': 'answer'})
    alpaca_4 = alpaca_4.filter(lambda example: len(example['input']) == 0)
    alpaca_4 = alpaca_4.remove_columns(['text', 'input'])
    alpaca_4.save_to_disk('raw_datasets/alpaca_4_only_qa')

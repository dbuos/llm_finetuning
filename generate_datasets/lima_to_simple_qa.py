from datasets import load_dataset

if __name__ == '__main__':
    lima = load_dataset("GAIR/lima")
    lima_train = lima['train'].filter(lambda x: len(x['conversations']) == 2)
    lima_qa = lima_train.map(lambda x: {"question": x['conversations'][0], "answer": x['conversations'][1]},
                             remove_columns=['conversations', 'source'])
    lima_qa.to_json('raw_datasets/lima_only_qa/qa_lima.json', orient='records', indent=4, lines=False)

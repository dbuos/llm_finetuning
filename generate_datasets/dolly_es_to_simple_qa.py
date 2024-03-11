from datasets import load_dataset

if __name__ == '__main__':
    dolly_es = load_dataset("argilla/databricks-dolly-15k-es-deepl")
    dolly_es = dolly_es['train'].filter(lambda x: len(x['context']) == 0)
    dolly_es = dolly_es.map(lambda x: {'question': x['instruction'], 'answer': x['response']})
    dolly_es = dolly_es.select_columns(['question', 'answer'])
    dolly_es.save_to_disk('raw_datasets/dolly_es_only_qa')

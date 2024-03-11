from uuid import uuid4

import datasets
from datasets import load_from_disk
from streaming import MDSWriter

# TODO: Implement answer template for more complex interactions, also implement multi-turn interactions preparation
TEMPLATES = {
    "long_instruction_prompt": (
        f"Debes responder a la siguiente pregunta propia del contexto de tecnología de Bancolombia:\n{{question}}\n---"
        "\nRespuesta:\n"),
    "short_instruction_prompt": f"[Bancolombia Dev User]:\n{{question}}\n[Respuesta]:\n",
    "short_instr_02": f"<|User|>\n{{question}}\n<|Agent|>\n",
    "short_text2lf": f"<|SENTENCE|>\n{{question}}\n<|LOGICAL FORM|>\n",
}


def load_json_instruct_dataset(path):
    dataset = datasets.load_dataset("json", data_files=path)
    dataset = dataset["train"]
    return dataset


def apply_prompt_template_fn(prompt_template_name, source):
    prompt_template = TEMPLATES[prompt_template_name]

    def apply_prompt_template(sample):
        return {
            "prompt": prompt_template.format(question=sample["question"]),
            "model_answer": sample["answer"],
            "uuid": str(uuid4()),
            "source_dataset": source
        }

    return apply_prompt_template


def write_instruct_streaming_dataset(out_root, dataset):
    hashes = ['sha256', 'xxh64']
    compression = 'zstd'
    columns = {
        'uuid': 'str',
        'prompt': 'str',
        'model_answer': 'str',
        'source_dataset': 'str'
    }

    with MDSWriter(out=out_root, columns=columns, compression=compression, hashes=hashes) as out:
        for sample in dataset:
            out.write(sample)


def load_instructions_dataset(source_path: str):
    if source_path.endswith('.json'):
        dataset = load_json_instruct_dataset(source_path)
    else:
        dataset = load_from_disk(source_path)
    return dataset


def prepare_and_write_json_instruct(instruct_template, source_path, out_uri):
    dataset = load_instructions_dataset(source_path)
    prompt_template_fn = apply_prompt_template_fn(instruct_template, source_path)
    dataset = dataset.map(prompt_template_fn, remove_columns=list(dataset.features))
    write_instruct_streaming_dataset(out_uri, dataset)


def prepare_and_write_json_lm(source_path, out_uri):
    dataset = load_instructions_dataset(source_path)

    def apply_prompt_template(sample):
        return {
            "prompt": sample["cleaned_text"],
            "model_answer": "",
            "uuid": str(uuid4()),
            "source_dataset": source_path
        }

    dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))
    write_instruct_streaming_dataset(out_uri, dataset)

# 1. Export de wikibot (OK)
# 2. Data sintetica de la wiki, generado openai (OK)
# 3. Instrucciones conversacion y function calling

# 4. Texto plano de PDFs
# 5. Extraccion wiki en texto plano

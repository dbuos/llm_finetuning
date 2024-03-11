import fire
from marvin_recipes.data.preparation import prepare_and_write_json_instruct


def main(template_name: str, source: str, out_path: str):
    prepare_and_write_json_instruct(
        instruct_template=template_name,
        source_path=source,
        out_uri=out_path)


if __name__ == '__main__':
    fire.Fire(main)

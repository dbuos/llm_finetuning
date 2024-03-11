python src/marvin_recipes/scripts/prepare_instruct_dataset.py \
  --template_name="short_text2lf" \
  --source="raw_datasets/recogs_qa/qa_recogs.json" \
  --out_path="s3://some-bucket/nlu_instruct/recogs_pairs_2"

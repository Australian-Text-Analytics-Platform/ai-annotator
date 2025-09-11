  atapllmc classify batch \
    --dataset 'tests/example_dataset.csv' \
    --column 'text' \
    --out-dir './example_out2' \
    --provider openai \
    --model 'gpt-4.1-mini' \
    --technique few_shot \
    --user-schema 'tests/example_user_schema_fewshot.json' \
    --api-key <your-api-key>


#atapllmc classify batch --dataset 'tests/example_dataset.csv' --column 'text' --out-dir './example_out2' --provider openai --model 'gpt-4.1-mini' --technique few_shot --user-schema 'tests/example_user_schema_fewshot.json' --api-key 
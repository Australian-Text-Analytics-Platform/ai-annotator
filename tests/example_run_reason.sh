  atapllmc classify batch \
    --dataset 'tests/example_dataset.csv' \
    --column 'text' \
    --out-dir './example_out3' \
    --provider openai \
    --model 'gpt-4.1-mini' \
    --technique chain_of_thought \
    --user-schema 'tests/example_user_schema_reason.json' \
    --api-key <your-api-key>


#atapllmc classify batch --dataset 'tests/example_dataset.csv' --column 'text' --out-dir './example_out3' --provider openai --model 'gpt-4.1-mini' --technique chain_of_thought --user-schema 'tests/example_user_schema_reason.json' --api-key 
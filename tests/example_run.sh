atapllmc classify batch \
--dataset 'tests/example_dataset.csv' \
--column 'text' \
--out-dir './example_out' \
--provider openai \
--model 'gpt-4.1-mini' \
--technique zero_shot \
--user-schema 'tests/example_user_schema.json'  \   # this can also be a raw json.
--api-key <>


#atapllmc classify batch --dataset 'tests/example_dataset.csv' --column 'text' --out-dir './example_out' --provider openai --model 'gpt-4.1-mini' --technique zero_shot --user-schema 'tests/example_user_schema.json' --api-key <>
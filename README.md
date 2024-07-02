# ATAP LLM Classifier

This repository provides an easy-to-use notebook to leverge LLMs for classification.

## CLI

```shell
python3.11 -m venv .venv
source .venv/bin/activate

# pipx install poetry
poetry install

atapllmc classify batch --help
```

```shell
# example user schema - zeroshot
{
  "classes": [
    {
      "name": "class name 1",
      "description": "description of class name 1"
    },
    {
      "name": "class name 2",
      "description": "description of class name 2"
    },
  ]
}


# example - openai
atapllmc classify batch \
--dataset 'example.csv' \
--column 'text' \
--out-dir './out' \
--provider openai \
--model 'gpt-3.5-turbo' \
--technique zero_shot \
--user-schema 'user_schema.json'  \   # this can also be a raw json.
--api-key <your-api-key>

# example - ollama
atapllmc classify batch \
--dataset 'example.csv' \
--column 'text' \
--out-dir './out' \
--provider ollama \
--model 'llama3:8b' \
--technique zero_shot \
--user-schema 'user_schema.json'    # this can also be a raw json.
# --endpoint <custom endpoint else http://127.0.0.1:11434 (see assets/providers.yml)>
```

## Notebook

Under development - update internal api from refactor

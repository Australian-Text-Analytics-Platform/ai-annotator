# AI Annotator

This repository provides an AI annotator tool for the LDaCA/ATAP Platform.


## Installation 

```shell
python3.11 -m venv .venv
source .venv/bin/activate

# pipx install poetry
poetry install

atapllmc classify batch --help
```

## How To - CLI

Create a user schema and save as json.
Example user schema - zeroshot:

```json
{
  "classes": [
    {
      "name": "class name 1",
      "description": "description of class name 1"
    },
    {
      "name": "class name 2",
      "description": "description of class name 2"
    }
  ]
}
```

# Example - OpenAI

Given a json user schema and a corpus csv file, run fhe following command below.
Include your OpenAI API key.

```shell
atapllmc classify batch \
--dataset 'example.csv' \
--column 'text' \
--out-dir './out' \
--provider openai \
--model 'gpt-4.1-mini' \
--technique zero_shot \
--user-schema 'user_schema.json'  \   # this can also be a raw json.
--api-key <your-api-key>
```

# Example - Ollama

```shell
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
# Contributing Guide for developers

## Overview

**Important! - This guide just give general directions to make it easier. Please read the code.**

Main thread code is basically in `pipeline.py`.

## Mock mode

Use mock mode in general when developing.

To enable mock mode:

```python
from atap_llm_classifier import config

config.mock.enabled = True
```

## Providers

#### Supported:

1. OpenAI
2. Ollama
3. SIH OpenAI (not implemented yet - see enum)

### Extending

1. Go to `providers/provider.py`
2. Add an enum
3. Add a key in `assets/providers.yml` corresponding to enum value
4. Follow pattern in the `provider.py` and `assets/providers.yml` files.

## Techniques

#### Supported:

1. ZeroShot
2. Chain of Thought (not implemented yet - see `techniques/cot.py`)

### Extending

1. Go to `techniques/techniques.py`
2. Add an enum
3. Add a key in `assets/techniques` corresponding to enum value
4. Create a subclass of `BaseTechnique`
5. Add your subclass to `Technique.prompt_maker_cls` in `techniqus/techniques.py`

## Modifiers

#### Supported:

1. NoModifier (default)
2. SelfConsistency (not implemented yet - see `modifiers/self_consistency.py` )

#### Extending

Pretty much the same pattern.

## Output Formats:

Also see `settings.py`, so you can supply env var `LLM_OUTPUT_FORMAT` to change it.

#### Supported:

1. JSON (default)
2. YAML

The default format is JSON for better reliability with LLM outputs. You can change this by setting `LLM_OUTPUT_FORMAT=yaml` in your environment or `.env` file.

#### Extending:

Pretty much the same pattern.
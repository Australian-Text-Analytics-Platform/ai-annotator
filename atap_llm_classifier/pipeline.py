"""pipeline.py

Defines the steps of the full pipeline.

The output
"""

import asyncio

import litellm
from pydantic import BaseModel, Field
from loguru import logger

from atap_corpus import Corpus
from atap_corpus._types import Doc
from atap_llm_classifier.core import LLMModelConfig
from atap_llm_classifier.providers import LLMProvider
from atap_llm_classifier.techniques import Technique
from atap_llm_classifier.modifiers import Modifier
from atap_llm_classifier.utils import timeit

litellm.set_verbose = False

MODEL = "ollama_chat/llama3"
API_BASE = "http://localhost:11434"
NUM_MESSAGES = 10


class PipelineResults(BaseModel):
    pass


def run(
    corpus: Corpus,
    provider: LLMProvider,
    technique: Technique,
    modifier: Modifier | None = None,
) -> PipelineResults:
    # runs

    return PipelineResults()

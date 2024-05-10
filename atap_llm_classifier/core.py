"""completions.py

This depends heavily on the litellm package.

Provide the functions to call the LLM.
"""

from pydantic import BaseModel

from atap_corpus import Corpus

from atap_llm_classifier.modifiers import Modifier
from atap_llm_classifier.techniques import Technique


class LLMArgs(BaseModel):
    model: str
    temperature: float = 1.0
    top_p: float = 1.0
    n_completions: int = 1


async def a_batch_classify(
    corpus: Corpus,
    result_col: str,
    llm_args: LLMArgs,
    technique: Technique,
    modifier: Modifier,
):
    pass
    # todo: modifier patterns below:
    # modifier -> apply_preconditions(llm_args)
    # modifier -> post_classification_callback(results)

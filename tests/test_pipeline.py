from atap_corpus import Corpus

from atap_llm_classifier import config, pipeline, models
from atap_llm_classifier.modifiers import Modifier
from atap_llm_classifier.providers import LLMProvider
from atap_llm_classifier.techniques import Technique, schemas


def test_batch():
    config.mock.enabled = True
    config.mock.tokens_rate_limit = None
    config.mock.requests_rate_limit = None

    corpus: Corpus = Corpus([f"Test document {i}" for i in range(10)])
    model_props = LLMProvider.OPENAI.properties.get_model_props("gpt-3.5-turbo")

    res: pipeline.BatchResults = pipeline.batch(
        corpus=corpus,
        model_props=model_props,
        llm_config=models.LLMConfig(),
        technique=Technique.ZERO_SHOT,
        user_schema=schemas.ZeroShotUserSchema(
            classes=[
                schemas.ZeroShotClass(
                    name="class name", description="class description"
                )
            ]
        ),
        modifier=Modifier.NO_MODIFIER,
        on_result_callback=None,
    )
    assert len(res.successes) == len(corpus)

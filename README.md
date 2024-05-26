# ATAP LLM Classifier

This repository provides an easy-to-use notebook that leverge LLMs for classification.

+ 0-Shot classification
+ K-Shot classification (Chain-Of-Thought)

## Archived

#### Multiple sys role implementation

```python
# # todo: do not use this - not implemented.
# async def a_run_multi_llm(
#     corpus: Corpus,
#     models: Sequence[str],
#     api_keys: Sequence[SecretStr],
#     technique: Technique | None = None,
#     modifier: Modifier | None = None,
# ):
#     # todo: make model, api_key, sys prompt, a BaseModel
#     #   allow multiple sys prompts to be used.
#     #
#     # todo: this basically runs a_run len(models) times.
#     #   then, just add an extra column that takes highest classified.
#     tasks = list()
#     for model, api_key in zip(models, api_keys):
#         task: Future = asyncio.create_task(
#             core.a_classify(
#                 text=corpus[:1],  # todo: use the corpus docs
#                 model=model,
#                 api_key=api_key,
#                 llm_config=LLMConfig(seed=42),
#                 technique=technique,
#                 modifier=modifier,
#             )
#         )
#         tasks.append(task)
#     results = await asyncio.gather(*tasks)
#     return results
```
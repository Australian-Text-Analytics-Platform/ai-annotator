"""cot.py

Chain of Thought - allows for N number of 'shots'.
"""

import pydantic
from pydantic import BaseModel, field_validator
from pydantic_core.core_schema import ValidationInfo

from atap_llm_classifier.assets import Asset

from atap_llm_classifier.techniques.techniques import BaseTechnique

__all__ = [
    "CoTExample",
]


# todo: here get the template from assets.


class CoTExample(BaseModel):
    query: str
    classification: str
    reason: str


class CoTClass(BaseModel):
    name: str
    description: str


class CoTSchema(BaseModel):
    classes: list[CoTClass]
    examples: list[CoTExample]

    @field_validator("examples", mode="after")
    @classmethod
    def classes_are_defined_for_examples(
        cls, v: list[CoTExample], info: ValidationInfo
    ):
        uniq_classes = set(map(lambda c: c.name, info.data.get("classes")))
        uniq_ex_classes = set(map(lambda ex: ex.classification, v))
        missing = uniq_ex_classes - uniq_classes
        if len(missing) > 0:
            raise pydantic.ValidationError(
                f"Missing class but in example. {', '.join(list(missing))}"
            )
        return v


def make_prompt_head(schema: CoTSchema) -> str:
    ex_joined = "\n".join(
        map(
            lambda ex: Asset.TECH_TEMPLATES.get("").format(
                example=ex.query,
                classification=ex.classification,
            ),
            schema.examples,
        )
    )
    return ex_joined


class ChainOfThought(BaseTechnique):
    prompt_schema = CoTSchema

    def make_prompt(self, text: str) -> str:
        prompt: CoTSchema = self.prompt
        head: str = make_prompt_head(prompt)

    @property
    def examples(self) -> list[CoTExample]:
        return self.prompt.examples

    @property
    def classes(self) -> list[CoTClass]:
        return self.prompt.classes

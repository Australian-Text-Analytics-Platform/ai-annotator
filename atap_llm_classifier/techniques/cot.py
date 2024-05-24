"""cot.py

Chain of Thought - allows for N number of 'shots'.
"""

import pydantic
from pydantic import BaseModel, field_validator
from pydantic_core.core_schema import ValidationInfo

from atap_llm_classifier.techniques import BaseTechnique
from atap_llm_classifier.techniques.templates import CoTTemplate
from atap_llm_classifier.techniques.techniques import Technique
from atap_llm_classifier.techniques import parsers

__all__ = [
    "ChainOfThought",
]


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


def make_prompt_examples(user_schema: CoTSchema) -> str:
    return "\n".join(
        map(
            lambda ex: ChainOfThought.template.user_schema_templates.example.format(
                example=ex.query,
                classification=ex.classification,
            ),
            user_schema.examples,
        )
    )


def make_prompt_classes(user_schema: CoTSchema) -> str:
    return "\n".join(
        map(
            lambda c: ChainOfThought.template.user_schema_templates.clazz.format(
                name=c.name,
                description=c.description,
            ),
            user_schema.classes,
        ),
    )


class ChainOfThought(BaseTechnique):
    schema = CoTSchema
    template: CoTTemplate = Technique.CHAIN_OF_THOUGHT.template

    def make_prompt(self, text: str) -> str:
        examples: str = make_prompt_examples(user_schema=self.user_schema)
        classes: str = make_prompt_classes(user_schema=self.user_schema)
        output_format: str = parsers.make_output_format_from_settings(
            self.template.outputs_format
        )
        return self.template.structure.format(
            num_classes=len(self.classes),
            examples=examples,
            classes=classes,
            output_format=output_format,
            text=text,
        )

    @property
    def examples(self) -> list[CoTExample]:
        return self.user_schema.examples

    @property
    def classes(self) -> list[CoTClass]:
        return self.user_schema.classes

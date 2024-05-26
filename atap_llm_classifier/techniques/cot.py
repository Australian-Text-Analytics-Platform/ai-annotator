"""cot.py

Chain of Thought - allows for N number of 'shots'.
"""

from atap_llm_classifier.techniques import Technique, BaseTechnique
from atap_llm_classifier.techniques.schemas import (
    CoTPromptTemplate,
    CoTUserSchema,
    CoTClass,
    CoTExample,
)

__all__ = [
    "ChainOfThought",
]


def make_prompt_examples(user_schema: CoTUserSchema) -> str:
    return "\n".join(
        map(
            lambda ex: ChainOfThought.template.user_schema_templates.example.format(
                example=ex.query,
                classification=ex.classification,
            ),
            user_schema.examples,
        )
    )


def make_prompt_classes(user_schema: CoTUserSchema) -> str:
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
    schema = CoTUserSchema
    template: CoTPromptTemplate = Technique.CHAIN_OF_THOUGHT.prompt_template

    def make_prompt(self, text: str) -> str:
        examples: str = make_prompt_examples(user_schema=self.user_schema)
        classes: str = make_prompt_classes(user_schema=self.user_schema)
        return self.template.structure.format(
            num_classes=len(self.classes),
            examples=examples,
            classes=classes,
            text=text,
        )

    @property
    def examples(self) -> list[CoTExample]:
        return self.user_schema.examples

    @property
    def classes(self) -> list[CoTClass]:
        return self.user_schema.classes

"""fewshot.py

Few-Shot Learning - provides examples without requiring reasoning.
"""

from functools import cached_property

from classifier_fastapi.techniques import Technique, BaseTechnique
from classifier_fastapi.techniques.schemas import (
    FewShotPromptTemplate,
    FewShotUserSchema,
    FewShotClass,
    FewShotExample,
)

__all__ = [
    "FewShot",
]


def make_prompt_examples(user_schema: FewShotUserSchema) -> str:
    return "\n".join(
        map(
            lambda ex: FewShot.template.user_schema_templates.example.format(
                example=ex.query,
                classification=ex.classification,
            ),
            user_schema.examples,
        )
    )


def make_prompt_classes(user_schema: FewShotUserSchema) -> str:
    return "\n".join(
        map(
            lambda c: FewShot.template.user_schema_templates.clazz.format(
                name=c.name,
                description=c.description,
            ),
            user_schema.classes,
        ),
    )


class FewShot(BaseTechnique):
    schema: FewShotUserSchema = FewShotUserSchema
    template: FewShotPromptTemplate = Technique.FEW_SHOT.prompt_template

    def make_prompt(self, text: str) -> str:
        examples: str = make_prompt_examples(user_schema=self.user_schema)
        classes: str = make_prompt_classes(user_schema=self.user_schema)
        reasoning_instruction = self._get_reasoning_instruction()
        return self.template.structure.format(
            num_classes=len(self.classes),
            examples=examples,
            classes=classes,
            text=text,
            reasoning_instruction=reasoning_instruction,
        )

    @cached_property
    def output_keys(self) -> list[str]:
        keys = [self.template.output_classification_key] + self.template.additional_output_keys
        if self.enable_reasoning:
            keys.append("reasoning")
        return keys

    @property
    def examples(self) -> list[FewShotExample]:
        return self.user_schema.examples

    @property
    def classes(self) -> list[FewShotClass]:
        return self.user_schema.classes
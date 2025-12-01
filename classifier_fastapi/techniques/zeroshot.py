from functools import cached_property

from classifier_fastapi.techniques import (
    Technique,
    BaseTechnique,
    ZeroShotPromptTemplate,
)
from classifier_fastapi.techniques.schemas import (
    ZeroShotUserSchema,
    ZeroShotClass,
)

__all__ = [
    "ZeroShot",
]


def make_prompt_classes(user_schema: ZeroShotUserSchema) -> str:
    return "\n".join(
        map(
            lambda c: ZeroShot.template.user_schema_templates.clazz.format(
                name=c.name,
                description=c.description,
            ),
            user_schema.classes,
        ),
    )


class ZeroShot(BaseTechnique):
    schema: ZeroShotUserSchema = ZeroShotUserSchema
    template: ZeroShotPromptTemplate = Technique.ZERO_SHOT.prompt_template

    def make_prompt(self, text: str) -> str:
        classes: str = make_prompt_classes(user_schema=self.user_schema)
        reasoning_instruction = self._get_reasoning_instruction()
        return self.template.structure.format(
            num_classes=len(self.classes),
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
    def classes(self) -> list[ZeroShotClass]:
        return self.user_schema.classes

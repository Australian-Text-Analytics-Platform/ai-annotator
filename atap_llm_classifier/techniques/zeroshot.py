from atap_llm_classifier.techniques import (
    Technique,
    BaseTechnique,
    ZeroShotPromptTemplate,
)
from atap_llm_classifier.techniques.schemas import (
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
        return self.template.structure.format(
            num_classes=len(self.classes),
            classes=classes,
            text=text,
        )

    @property
    def classes(self) -> list[ZeroShotClass]:
        return self.user_schema.classes

from pydantic import BaseModel

from atap_llm_classifier import Technique
from atap_llm_classifier.techniques import BaseTechnique
from atap_llm_classifier.techniques.schemas import ZeroShotPromptTemplate

__all__ = [
    "ZeroShot",
]

template: ZeroShotPromptTemplate = Technique.ZERO_SHOT.template


class ZeroShotClass(BaseModel):
    name: str
    description: str


class ZeroShotUserSchema(BaseModel):
    classes: list[ZeroShotClass]


def make_prompt_classes(user_schema: ZeroShotUserSchema) -> str:
    return "\n".join(
        map(
            lambda c: template.user_schema_templates.clazz.format(
                name=c.name,
                description=c.description,
            ),
            user_schema.classes,
        ),
    )


class ZeroShot(BaseTechnique):
    schema = ZeroShotUserSchema
    template = Technique.ZERO_SHOT.template

    def make_prompt(self, text: str) -> str:
        classes: str = make_prompt_classes(user_schema=self.user_schema)
        return template.structure.format(
            num_classes=len(self.classes),
            classes=classes,
            text=text,
        )

    @property
    def classes(self) -> list[ZeroShotClass]:
        return self.user_schema.classes

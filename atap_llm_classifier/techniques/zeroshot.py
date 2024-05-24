from pydantic import BaseModel

from atap_llm_classifier import Technique
from atap_llm_classifier.techniques import BaseTechnique, parsers
from atap_llm_classifier.techniques.templates import ZeroShotTemplate

__all__ = [
    "ZeroShot",
]

template: ZeroShotTemplate = Technique.ZERO_SHOT.template


class ZeroShotClass(BaseModel):
    name: str
    description: str


class ZeroShotSchema(BaseModel):
    classes: list[ZeroShotClass]


def make_prompt_classes(user_schema: ZeroShotSchema) -> str:
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
    schema = ZeroShotSchema
    template = Technique.ZERO_SHOT.template

    def make_prompt(self, text: str) -> str:
        classes: str = make_prompt_classes(user_schema=self.user_schema)
        output_format: str = parsers.make_output_format_from_settings(
            template.output_formats
        )
        return template.structure.format(
            num_classes=len(self.classes),
            classes=classes,
            output_format=output_format,
            text=text,
        )

    @property
    def classes(self) -> list[ZeroShotClass]:
        return self.user_schema.classes

import enum
import json
from pydantic import BaseModel, Field


class LLMModelConfig(BaseModel):
    temperature: float = 1.0
    top_p: float = 1.0
    n_completions: int = Field(1, gt=0)
    seed: int | None = Field(42)  # reproducible on default
    # todo: presence_penalty, frequenc_penalty, logit_bias, max_tokens


@enum.unique
class LiteLLMRole(enum.StrEnum):
    USER: str = "user"
    SYS: str = "system"
    ASSISTANT: str = "assistant"


class LiteLLMMessage(BaseModel):
    content: str = Field(frozen=True)
    role: LiteLLMRole


class LiteLLMArgs(BaseModel):
    model: str
    messages: list[LiteLLMMessage]
    stream: bool = False
    temperature: float
    top_p: float
    n: int
    api_key: str

    def to_fn_args(self):
        # note: so Enums are converted to str. - just being lazy.
        return json.loads(self.model_dump_json())

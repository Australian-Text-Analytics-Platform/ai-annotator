import enum
import json
from pydantic import BaseModel, Field, ConfigDict, field_validator


class LLMConfig(BaseModel):
    temperature: float = Field(1.0, gt=0)
    top_p: float = Field(1.0, gt=0, le=1.0)
    n_completions: int = Field(1, gt=0)
    seed: int | None = Field(42)  # reproducible on default
    reasoning_effort: str | None = Field(None)  # "low", "medium", "high", or None
    # todo: presence_penalty, frequenc_penalty, logit_bias, max_tokens

    @field_validator("reasoning_effort", mode="after")
    @classmethod
    def validate_reasoning_effort(cls, v: str | None):
        if v is not None and v not in ["low", "medium", "high"]:
            raise ValueError("reasoning_effort must be 'low', 'medium', 'high', or None")
        return v


@enum.unique
class LiteLLMRole(enum.StrEnum):
    USER: str = "user"
    SYS: str = "system"
    ASSISTANT: str = "assistant"


class LiteLLMMessage(BaseModel):
    content: str = Field(frozen=True)
    role: LiteLLMRole


class LiteLLMCompletionArgs(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[LiteLLMMessage]
    stream: bool = False
    temperature: float
    top_p: float
    n: int
    api_key: str | None = None
    base_url: str | None = None
    reasoning_effort: str | None = None

    def to_kwargs(self) -> dict:
        # note: so that Enums are converted to str. - just being lazy.
        kwargs = json.loads(self.model_dump_json())

        # Anthropic models don't support both temperature and top_p at the same time
        # Remove top_p for Anthropic models (claude-* models)
        if self.model and 'claude' in self.model.lower():
            kwargs.pop('top_p', None)

        # Remove reasoning_effort if None to avoid passing it to LiteLLM unnecessarily
        if kwargs.get('reasoning_effort') is None:
            kwargs.pop('reasoning_effort', None)

        return kwargs

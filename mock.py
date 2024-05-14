from pydantic import BaseModel


class Mock(BaseModel):
    response: str = "a mock response"
    model: str = "gpt-3.5-turbo"

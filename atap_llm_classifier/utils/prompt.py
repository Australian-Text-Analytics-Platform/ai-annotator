"""prompt_utils.py"""

from typing import Protocol, Any

import tiktoken


class TokenEncoder(Protocol):
    def encode(self, text: str) -> list[Any]:
        """Tokenises the text into token encodings."""
        pass


def get_token_encoder_for_openai(model: str) -> TokenEncoder:
    return tiktoken.encoding_for_model(model)

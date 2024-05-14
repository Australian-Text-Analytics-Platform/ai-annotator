"""cot.py

Chain of Thought - allows for N number of 'shots'.
"""

from .techniques import BaseTechnique


class ChainOfThought(BaseTechnique):
    def make_prompt(self, prompt: str) -> str:
        # todo: apply the chain of thought to the prompt
        return prompt

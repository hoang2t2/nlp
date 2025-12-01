import re
from src.core.interfaces import Tokenizer

class SimpleTokenizer(Tokenizer):
    def tokenize(self, text: str) -> list[str]:
        text = text.lower()

        for char in [".", ",", "?", "!"]:
            text = text.replace(char, f" {char} ")
        tokens = text.split()
        return tokens

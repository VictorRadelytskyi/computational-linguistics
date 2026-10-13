"""Whitespace-and-punctuation tokenizer for the tokenization benchmark."""

from collections import Counter
import re
from typing import Iterable


UNK_TOKEN = "<UNK>"
DEFAULT_VOCAB_SIZE = 50_257

_TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_WORD_PATTERN = re.compile(r"\w+", re.UNICODE)


class WhitespaceTokenizer:
    """Split text into words and punctuation, mapping unseen tokens to ``<UNK>``."""

    def __init__(self, vocab_size: int = DEFAULT_VOCAB_SIZE) -> None:
        if vocab_size < 1:
            raise ValueError("vocab_size must be at least 1")
        self.vocab_size = vocab_size
        self._token_to_id: dict[str, int] = {UNK_TOKEN: 0}
        self._fitted = False

    def fit(self, texts: Iterable[str]) -> "WhitespaceTokenizer":
        """Build a frequency-ranked vocabulary from the training texts."""
        counts: Counter[str] = Counter()
        for text in texts:
            if not isinstance(text, str):
                raise TypeError("texts must contain only strings")
            counts.update(_TOKEN_PATTERN.findall(text))

        tokens = sorted(counts, key=lambda token: (-counts[token], token))
        tokens = [token for token in tokens if token != UNK_TOKEN]
        tokens = tokens[: self.vocab_size - 1]
        self._token_to_id = {UNK_TOKEN: 0}
        self._token_to_id.update(
            (token, index) for index, token in enumerate(tokens, start=1)
        )
        self._fitted = True
        return self

    @property
    def vocabulary(self) -> tuple[str, ...]:
        """Return vocabulary tokens in ID order."""
        self._require_fitted()
        return tuple(
            token
            for token, _ in sorted(
                self._token_to_id.items(), key=lambda item: item[1]
            )
        )

    def tokenize(self, text: str) -> list[str]:
        """Return known tokens, replacing out-of-vocabulary tokens with ``<UNK>``."""
        self._require_fitted()
        return [
            token if token in self._token_to_id else UNK_TOKEN
            for token in _TOKEN_PATTERN.findall(text)
        ]

    def encode(self, text: str) -> list[int]:
        """Convert text to token IDs."""
        return [self._token_to_id[token] for token in self.tokenize(text)]

    def word_oov_statistics(self, text: str) -> dict[str, int | float]:
        """Count OOV words, excluding punctuation from the word-level denominator."""
        self._require_fitted()
        words = _WORD_PATTERN.findall(text)
        oov_count = sum(word not in self._token_to_id for word in words)
        total_count = len(words)
        return {
            "total_words": total_count,
            "oov_words": oov_count,
            "oov_percentage": (
                100.0 * oov_count / total_count if total_count else 0.0
            ),
        }

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("fit() must be called before using the tokenizer")

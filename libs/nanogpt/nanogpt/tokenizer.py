"""Char-level tokenizer: build vocab, encode/decode."""
import json
from pathlib import Path


class CharTokenizer:
    """Character-level tokenizer. The vocabulary is the set of unique
    characters in the training corpus; each maps to an integer ID."""
    def __init__(self, chars: list[str]):
        self.chars = chars
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        # The vocab IS the unique characters in the corpus.
        # sorted() is load-bearing: it makes IDs deterministic, so the
        # same corpus always yields the same mapping across runs.
        return cls(sorted(set(text)))

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)
    
    def save(self,path: Path) -> None:
        path.write_text(json.dumps({"chars": self.chars}), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "CharTokenizer":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(data["chars"])

if __name__ == "__main__":
    text = Path("data/processed/sample.txt").read_text(encoding="utf-8")
    tok = CharTokenizer.from_text(text)
    assert tok.decode(tok.encode(text)) == text, "round-trip failed!"
    print(f"vocab_size: {tok.vocab_size}")
    print(f"corpus length: {len(text)} chars")
    print(f"vocab: {''.join(tok.chars)}")
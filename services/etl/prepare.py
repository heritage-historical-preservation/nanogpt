"""ETL: clean raw corpus text and emit the processed corpus + tokenizer meta.

Runs in two places with the same code:
- Locally:            python services/etl/prepare.py            (data/raw -> data/processed)
- SageMaker Processing: the job mounts S3 inputs/outputs at
  /opt/ml/processing/{input,output} and runs this script with those dirs.

The transform is per-corpus cleaning for the source text (a 1913 scan with
page markers, footnote anchors, and editorial brackets).
"""

import argparse
import re
from pathlib import Path

from nanogpt.tokenizer import CharTokenizer


def strip_page_markers(text: str) -> str:
    # Remove [p. 123] / [p. xiv] style page references.
    # \[p\. \s* digits-or-roman \] — keep it tight so it can't eat real text.
    return re.sub(r"\[p\.\s*[\divxlcdm]+\]", "", text, flags=re.IGNORECASE)


def strip_formatting_chars(text: str) -> str:
    # The <, >, ^, ` you identified as formatting, not content.
    return re.sub(r"[<>^`]", "", text)


def normalize_whitespace(text: str) -> str:
    # Collapse runs of blank lines to at most two (paragraph breaks),
    # and trim trailing spaces on each line.
    text = re.sub(r"[ \t]+\n", "\n", text)         # trailing spaces
    text = re.sub(r"\n{3,}", "\n\n", text)         # 3+ newlines -> 2
    return text

def strip_editorial_brackets(text: str) -> str:
    # Footnote anchors: [*], [**], [*+], [*++], etc. — brackets containing
    # only * and + symbols.
    text = re.sub(r"\[[*+]+\]", "", text)
    # Note references: [n295] etc. — bracket, n, digits.
    text = re.sub(r"\[n\d+\]", "", text)
    # Editorial flow markers.
    text = re.sub(r"\[paragraph continues\]", "", text)
    # Copywrite date
    text = re.sub(r"\[1913\]", "", text)
    return text

def clean(text: str) -> str:
    # Order matters — see notes below.
    text = strip_page_markers(text)
    text = strip_formatting_chars(text)
    text = normalize_whitespace(text)
    text = strip_editorial_brackets(text)
    return text


def run(input_dir: Path, output_dir: Path) -> None:
    """Clean every .txt in input_dir into one corpus + tokenizer meta."""
    txt_files = sorted(input_dir.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"no .txt files found in {input_dir}")

    # Concatenate all inputs into a single corpus (double newline seam).
    raw = "\n\n".join(p.read_text(encoding="utf-8") for p in txt_files)
    cleaned = clean(raw)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "sample.txt").write_text(cleaned, encoding="utf-8")

    tok = CharTokenizer.from_text(cleaned)
    tok.save(output_dir / "meta.json")

    print(f"inputs:         {[p.name for p in txt_files]}")
    print(f"vocab_size:     {tok.vocab_size}")
    print(f"raw length:     {len(raw)}")
    print(f"cleaned length: {len(cleaned)}")
    print(f"removed:        {len(raw) - len(cleaned)} chars")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()
    run(args.input_dir, args.output_dir)

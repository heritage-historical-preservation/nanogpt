import re
from pathlib import Path

RAW_PATH = Path("data/raw/source.txt")
OUT_PATH = Path("data/processed/sample.txt")


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



if __name__ == "__main__":
    raw = RAW_PATH.read_text(encoding="utf-8")
    cleaned = clean(raw)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(cleaned, encoding="utf-8")

    print(f"raw length:     {len(raw)}")
    print(f"cleaned length: {len(cleaned)}")
    print(f"removed:        {len(raw) - len(cleaned)} chars")
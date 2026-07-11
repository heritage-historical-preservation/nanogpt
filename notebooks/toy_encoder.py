"""Toy tensor that actually does what makes a tensor a tensor:
flat memory + shape + strides => N-dimensional indexing.
Still pure Python, still slow, but now structurally honest."""

from math import prod

# --- 1. The tokenizer half (you already have this; reproduced minimally) ---

def build_vocab(text: str) -> list[str]:
    return sorted(set(text))


def encode(text: str, stoi: dict[str, int]) -> list[int]:
    # text -> flat list of integer IDs. This is the part torch is NOT doing.
    return [stoi[c] for c in text]


class ToyTensor:
    def __init__(self, data, shape, strides=None, dtype="int64"):
        self.data = data                      # flat list — the real memory
        self.shape = tuple(shape)             # how we INTERPRET it
        self.dtype = dtype
        # strides: how many flat steps to move 1 unit along each dim.
        # Default = "row-major"/C-contiguous: last dim is contiguous.
        self.strides = tuple(strides) if strides else self._contiguous_strides(shape)

    @staticmethod
    def _contiguous_strides(shape):
        # For shape (a, b, c): strides are (b*c, c, 1).
        # Moving one row means skipping a whole row's worth of elements.
        strides = []
        acc = 1
        for dim in reversed(shape):
            strides.insert(0, acc)
            acc *= dim
        return tuple(strides)

    def _flat_index(self, idx):
        # idx is a tuple like (row, col). Dot it with strides.
        return sum(i * s for i, s in zip(idx, self.strides))

    def __getitem__(self, idx):
        if isinstance(idx, int):
            idx = (idx,)
        return self.data[self._flat_index(idx)]

    def reshape(self, *new_shape):
        # Same data, new interpretation. NOTHING is copied or moved.
        assert prod(new_shape) == prod(self.shape), "element count must match"
        return ToyTensor(self.data, new_shape, dtype=self.dtype)

    def __repr__(self):
        return f"ToyTensor(shape={self.shape}, strides={self.strides}, data={self.data})"

    def to_grid_str(self) -> str:
        # REconstruct the N-D laout from flat data using shape + strides.
        # This is the inverse of what indexing does: instead of coords -> offset,
        # we walk all coordinate combinations and pull each value.

        if len(self.shape) == 1:
            # 1-D: Just a row.
            return "[" + "  ".join(str(self[i]) for i in range(self.shape[0])) + "]"
        
        if len(self.shape) == 2:
            rows, cols = self.shape
            # For each (r, c), look up via the stride dot product (self[r,c])
            lines = []
            for r in range(rows):
                cells = [str(self[r,c]) for c in range(cols)]
                # pad each cell to equal width so columns line up
                width = max(len(str(v)) for v in self.data)
                cells = [c.rjust(width) for c in cells]
                lines.append("[ " + "  ".join(cells) + " ]")
            return "\n".join(lines)
        
        return f"<{len(self.shape)}-D tensor, shape={self.shape}>"

if __name__ == "__main__":
    # One flat block of 12 numbers.
    # flat = list(range(12))   # [0,1,2,...,11]
    text = "the blue sky is vast and bright with beauty all around"
    chars = build_vocab(text)
    stoi = {ch: i for i , ch in enumerate(chars)}
    corpus = encode(text, stoi)
    t = ToyTensor(corpus, shape=(12,))
    print("1-D:", t, "\n")

    # Reinterpret the SAME 12 numbers as a 3x4 grid. No copy.
    g = t.reshape(3, 4)
    print("3x4:", g)
    print(g.to_grid_str())
    print("strides:", g.strides, "(move 4 in flat memory to go down a row)\n")
    i = t.reshape(4,3)
    print("\n4x3:", i)
    print(i.to_grid_str())

    # Index it 2-dimensionally:
    print("g[1, 2] =", g[1, 2], "-> flat index", g._flat_index((1, 2)))
    # row 1, col 2 -> 1*4 + 2*1 = flat[6] = 6

    # Reinterpret AGAIN as 2x6 — still the same flat memory.
    h = t.reshape(2, 6)
    print("\n2x6:", h)
    print(h.to_grid_str())
    print("g and h share data object:", g.data is h.data)  # True — no copy
    j = t.reshape(6,2)
    print("\n6x2:", j)
    print(j.to_grid_str())
    
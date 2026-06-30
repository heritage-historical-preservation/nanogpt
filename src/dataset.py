"""Batching: sample sequences from the token array, make x/y pairs.

y is x shifted by one token (next-token prediction targets).
"""


def get_batch(data, block_size, batch_size, device):
    raise NotImplementedError("sample random offsets, slice x and y, move to device")

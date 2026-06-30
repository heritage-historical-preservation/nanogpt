"""Training loop: forward, loss, backward, step.

Loads token arrays, builds the model, runs the optimization loop with periodic
eval on the val split, and saves checkpoints.
"""

from config import config


def main():
    raise NotImplementedError("load data, build model+optimizer, loop, eval, checkpoint")


if __name__ == "__main__":
    main()

import numba.core.types.containers
import numpy as np
import numba as nb
from numba.experimental import jitclass
# from numba.typed import List

# Numba's experimental Jitclasses require info on the attributes of the class
position_spec = [
    ("board", nb.uint8[:]),
    ("white_pieces", numba.types.List(nb.int64)),
    ("black_pieces", numba.types.List(nb.int64)),
    ("king_positions", nb.uint8[:]),
    ("castle_ability_bits", nb.uint8),
    ("ep_square", nb.int8),  # Cannot be u-ints because we do subtraction on it
    ("side", nb.uint8),
    ("hash_key", nb.uint64)
]


@jitclass(position_spec)
class Position:
    def __init__(self):
        self.board = np.zeros(120, dtype=np.int8)
        self.white_pieces = [nb.int64(1) for _ in range(0)]
        self.black_pieces = [nb.int64(1) for _ in range(0)]
        self.king_positions = np.zeros(2, dtype=np.uint8)
        self.castle_ability_bits = 0
        self.ep_square = 0
        self.side = 0
        self.hash_key = 0


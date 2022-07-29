
import numpy as np
import numba as nb
from numba.experimental import jitclass

# Numba's experimental Jitclasses require info on the attributes of the class
position_spec = [
    ("board", nb.int8[:]),  # Cannot be u-ints because subtraction of u-ints returns floats with numba
    ("white_king_position", nb.int8),
    ("black_king_position", nb.int8),
    ("castle_ability_bits", nb.uint8),
    ("ep_square", nb.int8),
    ("side", nb.int8),
    ("hash_key", nb.uint64)
]


@jitclass(position_spec)
class Position:
    def __init__(self):
        self.board = np.zeros(120, dtype=np.int8)
        self.white_king_position = 0
        self.black_king_position = 0
        self.castle_ability_bits = 0
        self.ep_square = 0
        self.side = 0
        self.hash_key = 0


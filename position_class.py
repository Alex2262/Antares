import numba.core.types.containers
import numpy as np
import numba as nb
from numba import njit
from numba.core import types
from numba.experimental import jitclass
from numba.experimental import structref

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
        self.board = np.zeros(120, dtype=np.uint8)
        self.white_pieces = [nb.int64(1) for _ in range(0)]
        self.black_pieces = [nb.int64(1) for _ in range(0)]
        self.king_positions = np.zeros(2, dtype=np.uint8)
        self.castle_ability_bits = 0
        self.ep_square = 0
        self.side = 0
        self.hash_key = 0


@structref.register
class PositionStructType(types.StructRef):
    def preprocess_fields(self, fields):
        return tuple((name, types.unliteral(typ)) for name, typ in fields)


class PositionStruct(structref.StructRefProxy):
    def __new__(cls, board,
                white_pieces,
                black_pieces,
                king_positions,
                castle_ability_bits,
                ep_square, side, hash_key):

        return structref.StructRefProxy.__new__(cls, board, white_pieces, black_pieces,
            king_positions, castle_ability_bits, ep_square, side, hash_key)

    @property
    def board(self):
        return PositionStruct_get_board(self)

    @property
    def white_pieces(self):
        return PositionStruct_get_white_pieces(self)

    @property
    def black_pieces(self):
        return PositionStruct_get_black_pieces(self)

    @property
    def king_positions(self):
        return PositionStruct_get_king_positions(self)

    @property
    def castle_ability_bits(self):
        return PositionStruct_get_castle_ability_bits(self)

    @property
    def ep_square(self):
        return PositionStruct_get_ep_square(self)

    @property
    def side(self):
        return PositionStruct_get_side(self)

    @property
    def hash_key(self):
        return PositionStruct_get_hash_key(self)


@njit
def PositionStruct_get_board(self):
    return self.board


@njit
def PositionStruct_get_white_pieces(self):
    return self.white_pieces


@njit
def PositionStruct_get_black_pieces(self):
    return self.black_pieces


@njit
def PositionStruct_get_king_positions(self):
    return self.king_positions


@njit
def PositionStruct_get_castle_ability_bits(self):
    return self.castle_ability_bits


@njit
def PositionStruct_get_ep_square(self):
    return self.ep_square


@njit
def PositionStruct_get_side(self):
    return self.side


@njit
def PositionStruct_get_hash_key(self):
    return self.hash_key


@njit
def PositionStruct_set_side(position, s):
    position.side = s


structref.define_proxy(PositionStruct, PositionStructType, ["board", "white_pieces", "black_pieces",
                                                            "king_positions", "castle_ability_bits",
                                                            "ep_square", "side", "hash_key"])


@njit(cache=True)
def init_position():
    position = PositionStruct(np.zeros(120, dtype=np.uint8),
                              [nb.int64(1) for _ in range(0)], [nb.int64(1) for _ in range(0)],
                              np.zeros(2, dtype=np.uint8), nb.uint8(0), nb.int8(0), nb.uint8(0), nb.uint64(0))

    return position


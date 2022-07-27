
import numpy as np
import numba as nb


np.random.seed(1)

MAX_HASH_SIZE       = 0x3640E2  # 32 mb
NO_HASH_ENTRY       = 2000000
USE_HASH_MOVE       = 3000000

NO_MOVE             = 0

INF                 = 1000000
MATE_SCORE          = 100000

ASPIRATION_VAL      = 50

# Search Constants
FULL_DEPTH_MOVES    = 3
REDUCTION_LIMIT     = 3
NULL_MOVE_REDUCTION = 2

# Piece Constants TODO: Refactor everything to use these constants

MOVE_TYPE_NORMAL    = 0
MOVE_TYPE_EP        = 1
MOVE_TYPE_CASTLE    = 2
MOVE_TYPE_PROMOTION = 3

WHITE_PAWN        = 0
WHITE_KNIGHT      = 1
WHITE_BISHOP      = 2
WHITE_ROOK        = 3
WHITE_QUEEN       = 4
WHITE_KING        = 5

BLACK_PAWN        = 6
BLACK_KNIGHT      = 7
BLACK_BISHOP      = 8
BLACK_ROOK        = 9
BLACK_QUEEN       = 10
BLACK_KING        = 11

EMPTY           = 12
PADDING         = 13

A1              = 91
A8              = 21
H1              = 98
H8              = 28

E1 = 95
E8 = 25
C1 = 93
C8 = 23
G1 = 97
G8 = 27


STANDARD_TO_MAILBOX = np.array((
    21, 22, 23, 24, 25, 26, 27, 28,
    31, 32, 33, 34, 35, 36, 37, 38,
    41, 42, 43, 44, 45, 46, 47, 48,
    51, 52, 53, 54, 55, 56, 57, 58,
    61, 62, 63, 64, 65, 66, 67, 68,
    71, 72, 73, 74, 75, 76, 77, 78,
    81, 82, 83, 84, 85, 86, 87, 88,
    91, 92, 93, 94, 95, 96, 97, 98,
))


MAILBOX_TO_STANDARD = np.array((
    99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
    99,  0,  1,  2,  3,  4,  5,  6,  7, 99,
    99,  8,  9, 10, 11, 12, 13, 14, 15, 99,
    99, 16, 17, 18, 19, 20, 21, 22, 23, 99,
    99, 24, 25, 26, 27, 28, 29, 30, 31, 99,
    99, 32, 33, 34, 35, 36, 37, 38, 39, 99,
    99, 40, 41, 42, 43, 44, 45, 46, 47, 99,
    99, 48, 49, 50, 51, 52, 53, 54, 55, 99,
    99, 56, 57, 58, 59, 60, 61, 62, 63, 99,
    99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99, 99, 99
))


WHITE_INCREMENTS = np.array((
    (-11,  -9, -10, -20,   0,   0,   0,   0),
    (-21, -19,  -8,  12,  21,  19,   8, -12),
    (-11,  11,   9,  -9,   0,   0,   0,   0),
    (-10,   1,  10,  -1,   0,   0,   0,   0),
    (-11,  11,   9,  -9, -10,   1,  10,  -1),
    (-11, -10,  -9,   1,  11,  10,   9,  -1)
))

BLACK_INCREMENTS = np.array((
    ( 11,   9,  10,  20,   0,   0,   0,   0),
    (-21, -19,  -8,  12,  21,  19,   8, -12),
    (-11,  11,   9,  -9,   0,   0,   0,   0),
    (-10,   1,  10,  -1,   0,   0,   0,   0),
    (-11,  11,   9,  -9, -10,   1,  10,  -1),
    (-11, -10,  -9,   1,  11,  10,   9,  -1)
))

WHITE_ATK_INCREMENTS = np.array((
    (-11,  -9,   0,   0,   0,   0,   0,   0),
    (-21, -19,  -8,  12,  21,  19,   8, -12),
    (-11,  11,   9,  -9,   0,   0,   0,   0),
    (-10,   1,  10,  -1,   0,   0,   0,   0),
    (-11,  11,   9,  -9, -10,   1,  10,  -1),
    (-11, -10,  -9,   1,  11,  10,   9,  -1)
))

BLACK_ATK_INCREMENTS = np.array((
    ( 11,   9,   0,   0,   0,   0,   0,   0),
    (-21, -19,  -8,  12,  21,  19,   8, -12),
    (-11,  11,   9,  -9,   0,   0,   0,   0),
    (-10,   1,  10,  -1,   0,   0,   0,   0),
    (-11,  11,   9,  -9, -10,   1,  10,  -1),
    (-11, -10,  -9,   1,  11,  10,   9,  -1)
))

PIECE_VALUES = np.array((82, 326, 352, 486, 982, 0))
ENDGAME_PIECE_VALUES = np.array((96, 292, 304, 512, 936, 0))

PST = np.array((
        (    0,   0,   0,   0,   0,   0,   0,   0,
            82, 120,  86,  95,  85, 110,  45,  27,
            10,  14,  17,  24,  24,  17,  14,  10,
             3,   4,  11,  16,  16,   9,   4,   3,
             0,  -2,  10,  15,  15,   3,   0,   0,
             2,   2,  -3,   4,   4,  -3,   2,   2,
             0,   0,   3, -26, -26,  12,   7,   0,
             0,   0,   0,   0,   0,   0,   0,   0),

        (  -90, -60, -30, -35,  -5, -30, -20, -80,
           -60, -20,  42,  22,  17,  36,   5, -40,
           -30,  42,  28,  46,  44,  78,   0,  15,
             0,   8,  23,  55,  58,  69,   8,   5,
           -10,   0,  13,  40,  42,  13,   0, -30,
           -30,   4,   4,  20,  23,   4,  12, -30,
           -40, -20,   1,   5,   5,   1, -20, -40,
           -80, -40, -30, -30, -30, -30, -40, -40),

        (  -20, -15, -10, -10, -10, -10, -15, -20,
           -15,   0,   0,   0,  10,  20,   0, -15,
           -10,  22,   5,  45,  28,  45,   0, -10,
           -10,  15,   5,  43,  35,  35,  15, -10,
           -10,  12,  15,  15,  18,  15,  12, -10,
           -10,  10,  10,   7,   7,  10,  10, -10,
           -10,  10,   0,   0,   0,   0,  10, -10,
           -20, -10, -10, -10, -10, -10, -10, -20),

        (   32,  40,  32,  45,  50,  10,  30,  35,
            27,  28,  38,  48,  60,  48,  28,  30,
            -5,  12,  12,  30,  18,  30,  12,   5,
           -20,  -5,  10,  15,  14,  20,  -5, -20,
           -30,  -5,  -1,   0,   5,   0,   0, -20,
           -35,  -5,  -2,  -1,   0,   0,  -2, -30,
           -30, -10,   5,   6,   6,   5,  -5, -40,
           -10,  -8,  10,  18,  18,  10, -20, -20),

        (  -20, -10, -10,  -5,  -5, -10, -10, -20,
           -10,  -6,   4,  -5,  -1,   6,   4, -10,
           -10,   0,   5,   5,   5,   5,   0, -10,
            -5,   0,   7,   5,   5,   5,   0,  -5,
            -5,   4,   8,   1,  -1,   2,   3,  -5,
           -10,  11,  11,  11,   8,  11,   5, -10,
           -10,  -2,   5,   0,   0,  -2,   0, -10,
           -20, -10, -10,  -5,  -5, -15, -10, -20),

        (  -30, -40, -40, -50, -50, -40, -40, -30,
           -30, -40, -40, -50, -50, -40, -40, -30,
           -30, -40, -40, -50, -50, -40, -40, -30,
           -30, -40, -40, -50, -50, -40, -40, -30,
           -20, -30, -30, -40, -40, -30, -30, -20,
           -10, -20, -20, -40, -40, -20, -20, -10,
            14,  18, -10, -55, -55, -15,  15,  14,
            21,  35,   3, -50,   0, -20,  32,  22),
))

ENDGAME_PST = np.array((
        (    0,   0,   0,   0,   0,   0,   0,   0,
           105, 106, 110, 108, 105, 100, 104, 105,
            80,  99,  80,  79,  69,  60,  79,  80,
             3,   4,   9,  16,  16,   9,   4,   3,
             0,  -2,   5,  15,  15,   5,   0,   0,
             2,   2,  -2,  -1,  -1,  -2,   2,   2,
            -5,   8,   3, -26, -26,   3,   8,  -5,
             0,   0,   0,   0,   0,   0,   0,   0),
        (  -60, -40, -30, -30, -30, -30, -40, -80,
           -40, -20,   0,   0,   0,   0, -20, -40,
           -30,   0,  20,  25,  25,  20,   0, -30,
           -30,   5,  25,  30,  30,  25,   5, -30,
           -30,   0,  25,  30,  30,  25,   0, -30,
           -30,   5,  20,  25,  25,  20,   5, -30,
           -40, -20,   0,   5,   5,   0, -20, -40,
           -30, -40, -30, -30, -30, -30, -40, -50),
        (  -20, -10, -10, -10, -10, -10, -10, -20,
           -10,   0,   0,   0,   0,   0,   0, -10,
           -10,   0,   5,  10,  10,   5,   0, -10,
           -10,  15,   5,  25,  25,   5,  15, -10,
           -10,   5,  20,  15,  15,  20,   5, -10,
           -10,  15,  15,  10,  10,  15,  15, -10,
           -10,   5,   0,   0,   0,   0,   5, -10,
           -20, -10, -10, -10, -10, -10, -10, -20),
        (   10,  10,  15,  15,  10,  10,   5,   5,
            20,  30,  33,  35,  35,  33,  30,  20,
             4,  18,  23,  25,  25,  23,  18,   4,
            -5,   0,   8,   8,   8,   8,   0,  -5,
            -5,   0,   0,   0,   0,   0,   0,  -5,
            -5,   0,   0,   0,   0,   0,   0,  -5,
            -5,   0,   5,   5,   5,   5,   0,  -5,
             0,   5,  10,  14,  14,  10,   5,   0),
        (  -20, -10, -10,  -5,  -5, -10, -10, -20,
           -10,   0,  30,  40,  60,  10,   0, -10,
           -10,   0,  20,  45,  50,  20,   0, -10,
            -5,   0,  10,  44,  54,  30,   0,  -5,
            -5,   0,  20,  44,  34,  20,   0,  -5,
           -10,   5,  20,  20,  20,  20,   5, -10,
           -10,   0,   5,   0,   0,   0,   0, -10,
           -20, -10, -10,  -5,  -5, -10, -10, -20),
        (    2,   8,  16,  14,  14,  16,   8,   2,
            14,  16,  20,  26,  26,  20,  16,  14,
            16,  24,  29,  31,  31,  29,  24,  16,
            16,  26,  32,  34,  34,  32,  26,  16,
             8,  24,  30,  33,  33,  30,  24,   8,
             2,   8,  16,  14,  14,  16,   8,   2,
           -18, -14, -10, -10, -10, -10, -14, -18,
           -20, -20, -20, -20, -20, -20, -20, -20),
))


PIECE_HASH_KEYS = np.random.randint(1, 2**64 - 1, size=(12, 64), dtype=np.uint64)
EP_HASH_KEYS = np.random.randint(1, 2**64 - 1, size=64, dtype=np.uint64)
CASTLE_HASH_KEYS = np.random.randint(1, 2 ** 64 - 1, size=16, dtype=np.uint64)
SIDE_HASH_KEY = np.random.randint(1, 2 ** 64 - 1, dtype=np.uint64)

# This allows for a structured array similar to a C struct
# This is 18 bytes
NUMBA_HASH_TYPE = nb.from_dtype(np.dtype(
    [("key", np.uint64), ("score", np.int32), ("flag", np.uint8), ("move", np.uint32), ("depth", np.int8)]
))

HASH_FLAG_EXACT, HASH_FLAG_ALPHA, HASH_FLAG_BETA = (0, 1, 2)


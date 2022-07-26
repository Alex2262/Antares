
import numpy as np
import numba as nb


np.random.seed(1)

MAX_HASH_SIZE       = 0x3640E2  # 64 mb
NO_HASH_ENTRY       = 2000000
USE_HASH_MOVE       = 3000000
REPETITION_TABLE_SIZE = 500

NO_MOVE             = 0

INF                 = 1000000
MATE_SCORE          = 100000

ASPIRATION_VAL      = 50

# Search Constants
FULL_DEPTH_MOVES    = 2
REDUCTION_LIMIT     = 3
FUTILITY_MIN_DEPTH = 2
FUTILITY_MARGIN_PER_DEPTH = 150

# Piece/Move/Position Constants
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

SCORE_TYPE = nb.int32
MOVE_TYPE = nb.uint32

# This allows for a structured array similar to a C struct
# This is 18 bytes
NUMBA_HASH_TYPE = nb.from_dtype(np.dtype(
    [("key", np.uint64), ("score", np.int32), ("flag", np.uint8), ("move", np.uint32), ("depth", np.int8)]
))

STANDARD_TO_MAILBOX = np.array((
    21, 22, 23, 24, 25, 26, 27, 28,
    31, 32, 33, 34, 35, 36, 37, 38,
    41, 42, 43, 44, 45, 46, 47, 48,
    51, 52, 53, 54, 55, 56, 57, 58,
    61, 62, 63, 64, 65, 66, 67, 68,
    71, 72, 73, 74, 75, 76, 77, 78,
    81, 82, 83, 84, 85, 86, 87, 88,
    91, 92, 93, 94, 95, 96, 97, 98
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
    (-11,  -9, -10, -20,   0,   0,   0,   0),  # Pawn
    (-21, -19,  -8,  12,  21,  19,   8, -12),  # Knight
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


GAME_PHASE_SCORES = np.array((0, 1, 1, 2, 4, 0))
PIECE_VALUES_MID = np.array((82, 326, 352, 486, 982, 0))  # I like even numbers :D especially 2 and 6
PIECE_VALUES_END = np.array((96, 292, 304, 512, 936, 0))

TEMPO_BONUS = 8

DOUBLED_PAWN_PENALTY_MID = 14
DOUBLED_PAWN_PENALTY_END = 20  # Doubled pawns are very easy to target in the endgame.

ISOLATED_PAWN_PENALTY_MID = 18
ISOLATED_PAWN_PENALTY_END = 12  # The person playing with the isolated pawns should trade off pieces.

BACKWARDS_PAWN_PENALTY_MID = 6
BACKWARDS_PAWN_PENALTY_END = 8  # Give this a higher base score, but we reduce multipliers in eval_pawn()

PASSED_PAWN_BONUS_MID = 9  # These are multiplied by the pawns row, so the base value shouldn't be too high.
PASSED_PAWN_BONUS_END = 17

BISHOP_PAIR_BONUS_MID = 55
BISHOP_PAIR_BONUS_END = 40

ROOK_SEMI_OPEN_FILE_BONUS_MID = 15
ROOK_SEMI_OPEN_FILE_BONUS_END = 20

ROOK_OPEN_FILE_BONUS_MID = 27
ROOK_OPEN_FILE_BONUS_END = 32

QUEEN_SEMI_OPEN_FILE_BONUS_MID = 5
QUEEN_SEMI_OPEN_FILE_BONUS_END = 8

QUEEN_OPEN_FILE_BONUS_MID = 10
QUEEN_OPEN_FILE_BONUS_END = 12


# Pawns in the center are good.
# Pawns near the king (generally the king side) are good.
# Pawns on the 7th rank do not get high PST values since they get past pawn bonuses
PAWN_PST_MID = np.array((   0,   0,   0,   0,   0,   0,   0,   0,
                           45,  50,  55,  60,  65,  55,  30,  10,  # passed: | 108, 113, 118, 123, 128, 118,  93,  73|
                           35,  40,  45,  50,  60,  45,  40,  25,  # passed: |  89,  94,  99, 104, 114,  99,  94,  79|
                            8,   9,  20,  25,  30,  20,   7,   3,
                            0,   0,  13,  18,  20,   8,   3,  -4,
                            2,   2,   0,   2,   4,  -5,  12,   0,
                            0,   0,   3, -26, -26,  12,  15,  -5,
                            0,   0,   0,   0,   0,   0,   0,   0))

# Pawns on the 6th and 7th rank are excellent, but they get past pawn bonuses rather than big PST scores.
# Let pawns stay on the second rank unless they can be pushed forwards.
# Pawns on the flank files are better when they are pushed more
# since they can become outside passed pawns
PAWN_PST_END = np.array((   0,   0,   0,   0,   0,   0,   0,   0,
                           75,  70,  60,  55,  55,  55,  65,  70,  # passed: | 194, 179, 179, 174, 174, 174, 184, 189|
                           55,  50,  45,  40,  40,  45,  50,  50,  # passed: | 157, 152, 147, 142, 142, 147, 152, 152|
                           30,  30,  20,  26,  26,  20,  25,  30,  # passed: | 115, 115, 105, 111, 111, 105, 110, 115|
                           10,   0,   5,   4,   4,   5,   0,   0,
                            2,   2,   0,   3,   3,   0,   2,   2,
                           10,  10,   5,   5,   5,   3,   1,   0,
                            0,   0,   0,   0,   0,   0,   0,   0))

KNIGHT_PST_MID = np.array(( -70, -60, -30, -35,  -5, -30, -20, -70,
                            -60,  -5,  40,  20,  20,  40,   5, -40,
                            -30,  30,  30,  45,  45,  70,  10,  15,
                              0,  10,  30,  50,  50,  60,  10,   5,
                            -10,   0,  15,  40,  40,  15,   0, -30,
                            -30,   5,  10,  20,  20,  10,  10, -30,
                            -40, -20,   1,   5,   5,   1, -20, -40,
                            -60, -40, -30, -30, -30, -20, -40, -40))

KNIGHT_PST_END = np.array((  -60, -40, -30, -30, -30, -30, -40, -80,
                             -40, -20,   0,   0,   0,   0, -20, -40,
                             -30,   0,  20,  25,  25,  20,   0, -30,
                             -30,   5,  25,  30,  30,  25,   5, -30,
                             -30,   0,  25,  30,  30,  25,   0, -30,
                             -30,   5,  20,  25,  25,  20,   5, -30,
                             -40, -20,   0,   5,   5,   0, -20, -40,
                             -30, -40, -30, -30, -30, -30, -40, -50))

BISHOP_PST_MID = np.array((  -20, -15, -10, -10, -10, -10, -15, -20,
                             -15,   0,   0,   5,  10,  20,   0, -15,
                             -10,  20,   5,  45,  30,  45,   0, -10,
                             -10,  15,   5,  45,  35,  35,  15, -10,
                             -10,  12,  15,  15,  15,  15,  12, -10,
                             -10,  10,  10,   7,   7,  10,  10, -10,
                             -10,  10,   0,   0,   0,   0,  10, -10,
                             -20, -10, -10, -10, -10, -10, -10, -20))

BISHOP_PST_END = np.array((  -20, -10, -10, -10, -10, -10, -10, -20,
                             -10,   0,   0,   0,   0,   0,   0, -10,
                             -10,   0,   5,  10,  10,   5,   0, -10,
                             -10,  15,   5,  25,  25,   5,  15, -10,
                             -10,   5,  20,  15,  15,  20,   5, -10,
                             -10,  15,  15,  10,  10,  15,  15, -10,
                             -10,   5,   0,   0,   0,   0,   5, -10,
                             -20, -10, -10, -10, -10, -10, -10, -20))

ROOK_PST_MID = np.array((  30,  30,  30,  35,  35,  30,  30,  35,
                           25,  30,  40,  40,  45,  40,  30,  30,
                            5,  10,  10,  30,  20,  30,  10,   5,
                          -20,  -5,  10,  15,  15,  20,  -5, -20,
                          -30,  -5,  -1,   0,   5,  -1,  -5, -20,
                          -35,   0,   0,   0,   0,   0,   0, -30,
                          -30, -10,   4,   6,   6,   4,  -5, -40,
                          -10,  -8,   8,  10,  10,   8, -15, -15))

ROOK_PST_END = np.array((  10,  10,  15,  15,  10,  10,   5,   5,
                           20,  30,  33,  35,  35,  33,  30,  20,
                            4,  18,  23,  25,  25,  23,  18,   4,
                           -5,   0,   8,   8,   8,   8,   0,  -5,
                           -5,   0,   0,   0,   0,   0,   0,  -5,
                           -5,   0,   0,   0,   0,   0,   0,  -5,
                           -5,   0,   5,   5,   5,   5,   0,  -5,
                            0,   5,  10,  14,  14,  10,   5,   0))

QUEEN_PST_MID = np.array(( -20, -10, -10,  -5,  -5, -10, -10, -20,
                           -10,  -5,   5,  -5,  -1,   5,   5, -10,
                           -10,   0,   5,   5,   5,   5,   0, -10,
                            -5,   0,  10,   5,   5,   5,   0,  -5,
                            -5,   5,  10,   1,  -1,   5,   5,  -5,
                           -10,  15,  15,  15,  15,  15,  10, -10,
                           -10,  -2,   5,   0,   0,  -2,   0, -10,
                           -20, -10, -10,  -5,  -5, -15, -10, -20))

QUEEN_PST_END = np.array(( -20, -10, -10,  -5,  -5, -10, -10, -20,
                           -10,   0,  30,  40,  60,  10,   0, -10,
                           -10,   0,  20,  45,  50,  20,   0, -10,
                            -5,   0,  10,  45,  55,  30,   0,  -5,
                            -5,   0,  20,  45,  35,  20,   0,  -5,
                           -10,   5,  20,  20,  20,  20,   5, -10,
                           -10,   0,   5,   0,   0,   5,   0, -10,
                           -20, -10, -10,  -5,  -5, -10, -10, -20))

KING_PST_MID = np.array(( -30, -40, -40, -50, -50, -40, -40, -30,
                          -30, -40, -40, -50, -50, -40, -40, -30,
                          -30, -40, -40, -50, -50, -40, -40, -30,
                          -30, -40, -40, -50, -50, -40, -40, -30,
                          -20, -30, -30, -40, -40, -30, -30, -20,
                          -10, -20, -20, -40, -40, -20, -20, -10,
                           10,  12, -10, -55, -55, -15,  14,  13,
                           19,  25,   3, -30,  -5, -20,  27,  22))

KING_PST_END = np.array((   2,   8,  16,  14,  14,  16,   8,   2,
                           14,  16,  20,  26,  26,  20,  16,  14,
                           16,  25,  30,  31,  31,  30,  25,  16,
                           16,  26,  32,  35,  35,  32,  26,  16,
                            8,  25,  30,  33,  33,  30,  25,   8,
                            2,   8,  16,  14,  14,  16,   8,   2,
                          -18, -14, -10, -10, -10, -10, -14, -18,
                          -20, -20, -20, -20, -20, -20, -20, -20))


PIECE_HASH_KEYS = np.random.randint(1, 2**64 - 1, size=(12, 64), dtype=np.uint64)
EP_HASH_KEYS = np.random.randint(1, 2**64 - 1, size=64, dtype=np.uint64)
CASTLE_HASH_KEYS = np.random.randint(1, 2 ** 64 - 1, size=16, dtype=np.uint64)
SIDE_HASH_KEY = np.random.randint(1, 2 ** 64 - 1, dtype=np.uint64)

HASH_FLAG_EXACT, HASH_FLAG_ALPHA, HASH_FLAG_BETA = (0, 1, 2)


MVV_LVA_TABLE = np.array((
        (105, 205, 305, 405, 505, 605),
        (104, 204, 304, 404, 504, 604),
        (103, 203, 303, 403, 503, 603),
        (102, 202, 302, 402, 502, 602),
        (101, 201, 301, 401, 501, 601),
        (100, 200, 300, 400, 500, 600),
    ))




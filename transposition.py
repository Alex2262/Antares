

import numpy as np
import numba as nb
from utilities import *


@nb.njit(cache=False)
def clear_transposition_table(transposition_table):
    for entry in transposition_table:
        entry.hash_key = 0
        entry.depth = 0
        entry.flag = 0
        entry.score = 0


def probe_tt_entry(engine, position, alpha, beta, depth):
    entry = engine.transposition_table[position.hash_key % MAX_HASH_SIZE]

    if entry.key == position.hash_key:
        if entry.depth >= depth:

            score = entry.score

            if entry.flag == HASH_FLAG_EXACT:
                return score
            if entry.flag == HASH_FLAG_ALPHA and entry.score <= alpha:
                return alpha
            if entry.flag == HASH_FLAG_BETA and entry.score >= beta:
                return beta

    return NO_HASH_ENTRY


def record_tt_entry(engine, position, score, flag, depth):
    index = position.hash_key % MAX_HASH_SIZE

    engine.transposition_table[index].key = position.hash_key
    engine.transposition_table[index].depth = depth
    engine.transposition_table[index].flag = flag
    engine.transposition_table[index].score = score




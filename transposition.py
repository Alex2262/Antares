

from utilities import *


@nb.njit(cache=True)
def probe_tt_entry(engine, position, alpha, beta, depth):
    entry = engine.transposition_table[position.hash_key % MAX_HASH_SIZE]

    if entry.key == position.hash_key:
        if entry.depth >= depth:

            score = entry.score

            if entry.flag == HASH_FLAG_EXACT:
                return score
            if entry.flag == HASH_FLAG_ALPHA and entry.score <= alpha:
                return score
            if entry.flag == HASH_FLAG_BETA and entry.score >= beta:
                return score

        return USE_HASH_MOVE + entry.move

    return NO_HASH_ENTRY


@nb.njit(cache=True)
def record_tt_entry(engine, position, score, flag, move, depth):
    index = position.hash_key % MAX_HASH_SIZE

    if engine.transposition_table[index].key != position.hash_key     \
            or depth > engine.transposition_table[index].depth        \
            or flag == HASH_FLAG_EXACT:

        engine.transposition_table[index].key = position.hash_key
        engine.transposition_table[index].depth = depth
        engine.transposition_table[index].flag = flag
        engine.transposition_table[index].score = score
        engine.transposition_table[index].move = move


@nb.njit(cache=True)
def probe_tt_entry_q(engine, position, alpha, beta):
    entry = engine.transposition_table[position.hash_key % MAX_HASH_SIZE]

    if entry.key == position.hash_key:
        score = entry.score

        if entry.flag == HASH_FLAG_EXACT:
            return score
        if entry.flag == HASH_FLAG_ALPHA and entry.score <= alpha:
            return score
        if entry.flag == HASH_FLAG_BETA and entry.score >= beta:
            return score
        return USE_HASH_MOVE + entry.move

    return NO_HASH_ENTRY


@nb.njit(cache=True)
def record_tt_entry_q(engine, position, score, flag, move):
    index = position.hash_key % MAX_HASH_SIZE

    if engine.transposition_table[index].key != position.hash_key:

        engine.transposition_table[index].key = position.hash_key
        engine.transposition_table[index].depth = -1
        engine.transposition_table[index].flag = flag
        engine.transposition_table[index].score = score
        engine.transposition_table[index].move = move

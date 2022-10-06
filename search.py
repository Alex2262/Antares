
"""
Antares V4 fully JIT compiled with Numba ~ 68k nps
Position class: contains board, castling ability, king position, en passant square
Board representation: 12x10 mailbox
Move representation: integer (28 bits used)

"""
import math
import time

from evaluation import *
from move_generator import *
from position import *
from transposition import *

from position_class import PositionStruct_set_side
from search_class import SearchStruct_set_max_depth, SearchStruct_set_max_time, SearchStruct_set_start_time, \
                         SearchStruct_set_current_search_depth


# @nb.njit(nb.float64(), cache=True)
@nb.njit(cache=True)
def get_time():
    with nb.objmode(time_amt=nb.float64):
        time_amt = time.time()

    return time_amt


# @nb.njit(nb.void(Search.class_type.instance_type), cache=True)
@nb.njit(cache=True)
def reset_search(engine):
    engine.node_count = 0

    engine.current_search_depth = 0
    engine.ply = 0

    engine.pv_table = np.zeros((engine.max_depth, engine.max_depth), dtype=np.uint32)
    engine.pv_length = np.zeros(engine.max_depth+1, dtype=np.uint16)

    engine.killer_moves = np.zeros((2, engine.max_depth), dtype=np.uint32)
    engine.history_moves = np.zeros((12, 64), dtype=np.uint32)

    engine.stopped = False


# @nb.njit(nb.void(Search.class_type.instance_type), cache=True)
@nb.njit(cache=True)
def new_game(engine):
    reset_search(engine)
    engine.repetition_table = np.zeros(REPETITION_TABLE_SIZE, dtype=np.uint64)
    engine.repetition_index = 0
    engine.transposition_table = np.zeros(MAX_HASH_SIZE, dtype=NUMBA_HASH_TYPE)


# @nb.njit(nb.void(Search.class_type.instance_type), cache=True)
@nb.njit(cache=True)
def update_search(engine):
    elapsed_time = 1000 * (get_time() - engine.start_time)
    if elapsed_time >= engine.max_time and engine.current_search_depth >= engine.min_depth:
        engine.stopped = True


# @nb.njit(nb.boolean(Search.class_type.instance_type, Position.class_type.instance_type), cache=True)
@nb.njit(cache=True)
def detect_repetition(engine, position):
    for i in range(engine.repetition_index-2, -1, -2):
        if engine.repetition_table[i] == position.hash_key:
            return True

    return False


# @nb.njit(SCORE_TYPE(Search.class_type.instance_type,
# |                    Position.class_type.instance_type, SCORE_TYPE, SCORE_TYPE, nb.int8))
@nb.njit(cache=False)
def qsearch(engine, position, alpha, beta, depth):

    # Update the search progress every 1024 nodes
    if engine.node_count & 1023 == 0:
        update_search(engine)

    if engine.stopped:
        return 0

    # Increase the node count
    engine.node_count += 1

    tt_value = probe_tt_entry_q(engine, position, alpha, beta)
    tt_move = NO_MOVE

    # A score was given to return
    if tt_value < NO_HASH_ENTRY:
        return tt_value

    # Use a tt entry move to sort moves
    elif tt_value > USE_HASH_MOVE:
        tt_move = tt_value - USE_HASH_MOVE

    # The stand pat:
    # We assume that in a position there should always be a quiet evaluation,
    # and we return this if it is better than the captures.
    static_eval = evaluate(position)

    # Beta is the alpha (the best evaluation) of the previous node
    if static_eval >= beta:
        return static_eval

    if depth == 0:
        return static_eval

    # Using a variable to record the hash flag
    tt_hash_flag = HASH_FLAG_ALPHA

    # Save info for undo move
    current_ep = position.ep_square
    current_hash_key = position.hash_key
    current_castle_ability_bits = position.castle_ability_bits

    # If our static evaluation has improved after the last move.
    alpha = max(alpha, static_eval)

    # Retrieving all pseudo legal captures as a list of [Move class]
    moves = get_pseudo_legal_captures(position)

    move_scores = get_capture_scores(moves, tt_move)

    best_score = static_eval
    best_move = NO_MOVE

    # Iterate through the noisy moves (captures) and search recursively with qsearch (quiescence search)
    for current_move_index in range(len(moves)):
        sort_next_move(moves, move_scores, current_move_index)
        move = moves[current_move_index]

        # Delta / Futility pruning
        # If the piece we capture plus a margin cannot even improve our score then
        # there is no point in searching it
        if static_eval + PIECE_VALUES_MID[get_occupied(move) % BLACK_PAWN] + 220 < alpha:
            continue

        # Make the capture
        attempt = make_move(position, move)
        if not attempt:
            undo_move(position, move, current_ep, current_castle_ability_bits, current_hash_key)
            continue

        position.side ^= 1

        return_eval = -qsearch(engine, position, -beta, -alpha, depth - 1)

        position.side ^= 1
        undo_move(position, move, current_ep, current_castle_ability_bits, current_hash_key)

        if engine.stopped:
            return 0

        if return_eval > best_score:
            best_score = return_eval
            best_move = move

            if return_eval > alpha:
                alpha = return_eval
                tt_hash_flag = HASH_FLAG_EXACT

                # beta cutoff
                if return_eval >= beta:
                    record_tt_entry_q(engine, position, best_score, HASH_FLAG_BETA, best_move)
                    return best_score

    record_tt_entry_q(engine, position, best_score, tt_hash_flag, best_move)

    return best_score


# @nb.njit(SCORE_TYPE(Search.class_type.instance_type,
# |                   Position.class_type.instance_type, SCORE_TYPE, SCORE_TYPE, nb.int8))
@nb.njit(cache=False)
def negamax(engine, position, alpha, beta, depth, do_null):

    # Initialize PV length
    engine.pv_length[engine.ply] = engine.ply

    # Detect repetitions after the first ply to allow for retrieving of moves
    # since we don't check for the full three-fold repetition.
    if engine.ply and detect_repetition(engine, position):
        return 0

    # Start quiescence search at the end of regular negamax search to counter the horizon effect.
    if depth == 0:
        return qsearch(engine, position, alpha, beta, engine.max_qdepth)

    # Increase node count after checking for terminal nodes since that would be counting double
    # nodes with quiescent search
    engine.node_count += 1

    # Terminate search when a condition has been reached, usually a time limit.
    if engine.stopped:
        return 0

    pv_node = alpha != beta - 1
    in_check = is_attacked(position,
                           position.king_positions[position.side])

    # Check extension
    if in_check:
        depth += 1

    # Get a value from probe_tt_entry that will correspond to either returning a score immediately
    # or returning no hash entry, or returning move int to sort
    tt_value = probe_tt_entry(engine, position, alpha, beta, depth)
    tt_move = NO_MOVE

    # A score was given to return
    if tt_value < NO_HASH_ENTRY:

        if not engine.ply:
            move = engine.transposition_table[position.hash_key % MAX_HASH_SIZE].move
            engine.pv_table[0][0] = move
            engine.pv_length[0] = 1

        return tt_value

    # Use a tt entry move to sort moves
    elif tt_value > USE_HASH_MOVE:
        tt_move = tt_value - USE_HASH_MOVE

    # Using a variable to record the hash flag
    tt_hash_flag = HASH_FLAG_ALPHA

    # Used at the end of the negamax function to determine checkmates, stalemates, etc.
    legal_moves = 0

    # Saving information to undo moves successfully.
    current_ep = position.ep_square
    current_hash_key = position.hash_key
    current_castle_ability_bits = position.castle_ability_bits

    # Reverse Futility Pruning
    if depth <= FUTILITY_MIN_DEPTH  \
            and not in_check        \
            and not pv_node:

        evaluation = evaluate(position)
        if evaluation - FUTILITY_MARGIN_PER_DEPTH * depth >= beta:
            return evaluation

    # Null move pruning
    # We give the opponent an extra move and if they are not able to make their position
    # any better, then our position is too good, and we don't need to search any deeper.
    if depth >= 3               \
            and do_null         \
            and not in_check    \
            and not pv_node:

        # Adaptive NMP
        # depth 3 == 2
        # depth 8 == 3
        # depth 13 == 4
        # depth 18 == 5
        reduction = (depth + 2)//5 + 1

        # Make the null move (flipping the position and clearing the ep square)
        make_null_move(position)

        engine.ply += 1
        # engine.repetition_index += 1
        # engine.repetition_table[engine.repetition_index] = position.hash_key

        # We will reduce the depth since the opponent gets two moves in a row to improve their position
        return_eval = -negamax(engine, position, -beta, -beta + 1, depth - 1 - reduction, False)
        engine.ply -= 1
        # engine.repetition_index -= 1

        # Undoing the null move (flipping the position back and resetting ep and hash)
        undo_null_move(position, current_ep, current_hash_key)

        if return_eval >= beta:
            return beta

    # Retrieving the pseudo legal moves in the current position as a list of integers
    # Score the moves
    moves = get_pseudo_legal_moves(position)
    move_scores = get_move_scores(engine, moves, tt_move)

    raised_alpha = False

    # Best move to save for TT
    best_move = NO_MOVE
    best_score = -INF

    # Iterate through moves and recursively search with Negamax
    for current_move_index in range(len(moves)):

        # Sort the next move. If an early move causes a cutoff then we have saved time
        # by only sorting one or a few moves rather than the whole list.
        sort_next_move(moves, move_scores, current_move_index)
        move = moves[current_move_index]

        if current_move_index == 0:
            best_move = move

        # Make the move
        attempt = make_move(position, move)

        # The move put us in check, therefore it was not legal, and we must disregard it
        if not attempt:
            undo_move(position, move, current_ep, current_castle_ability_bits, current_hash_key)
            continue

        # We proceed since the move is legal
        # Flip the position for the opposing player
        position.side ^= 1

        # increase ply and repetition index
        engine.ply += 1
        engine.repetition_index += 1
        engine.repetition_table[engine.repetition_index] = position.hash_key

        reduction = 0
        is_killer_move = move == engine.killer_moves[0][engine.ply - 1] or \
                         move == engine.killer_moves[1][engine.ply - 1]

        # Late Move Reductions (LMR)
        if legal_moves >= FULL_DEPTH_MOVES                                                  \
                and ((engine.ply and not pv_node) or legal_moves >= FULL_DEPTH_MOVES + 2)   \
                and depth >= REDUCTION_LIMIT                                                \
                and not in_check                                                            \
                and get_move_type(move) == 0                                                \
                and not get_is_capture(move):

            reduction = math.sqrt(depth) * 0.5 + math.sqrt(legal_moves) * 0.55 - 0.3

            reduction -= pv_node

            reduction -= is_killer_move

            reduction -= engine.history_moves[get_selected(move)][MAILBOX_TO_STANDARD[get_to_square(move)]] / 20000

            # We don't want to go straight to quiescence search from LMR.
            reduction = min(depth - 2, max(1, int(reduction)))

        # PVS
        if legal_moves == 0:
            return_eval = -negamax(engine, position, -beta, -alpha, depth - reduction - 1, True)
        else:
            return_eval = -negamax(engine, position, -alpha - 1, -alpha, depth - reduction - 1, True)

        # The move was actually good, so we can try a zero window search at full depth
        if return_eval > alpha and reduction and legal_moves != 0:
            return_eval = -negamax(engine, position, -alpha - 1, -alpha, depth - 1, True)

        # Either the full depth zero window search returned above alpha, or
        # The reduced alpha - beta window search returned above alpha
        if return_eval > alpha and reduction:
            return_eval = -negamax(engine, position, -beta, -alpha, depth - 1, True)

        # Decrease the ply and repetition index
        engine.ply -= 1
        engine.repetition_index -= 1

        # Flip the position back to us and undo the move
        position.side ^= 1
        undo_move(position, move, current_ep, current_castle_ability_bits, current_hash_key)

        # Prevent saving of any search termination scores
        if engine.stopped:
            return 0

        # The move is better than other moves searched
        if return_eval > best_score:
            best_score = return_eval

            # Write pv move
            engine.pv_table[engine.ply][engine.ply] = move
            for next_ply in range(engine.ply + 1, engine.pv_length[engine.ply + 1]):
                engine.pv_table[engine.ply][next_ply] = engine.pv_table[engine.ply + 1][next_ply]

            engine.pv_length[engine.ply] = engine.pv_length[engine.ply + 1]

            if return_eval > alpha:

                raised_alpha = True

                alpha = return_eval
                best_move = move

                # Exact flag reached
                tt_hash_flag = HASH_FLAG_EXACT

                # Store history moves
                if not get_is_capture(move):
                    engine.history_moves[get_selected(move)][MAILBOX_TO_STANDARD[get_to_square(move)]] += depth * depth

                # Alpha - Beta cutoff. This is a 'cut node' and we have failed high
                if return_eval >= beta:

                    # Store killer moves
                    if not get_is_capture(move):
                        engine.killer_moves[1][engine.ply] = engine.killer_moves[0][engine.ply]
                        engine.killer_moves[0][engine.ply] = move

                    if -MATE_SCORE < best_score < MATE_SCORE:
                        record_tt_entry(engine, position, best_score, HASH_FLAG_BETA, best_move, depth)

                    return best_score

        # Increase legal moves
        legal_moves += 1

    # Stalemate: No legal moves and king is not in check
    if legal_moves == 0 and not in_check:
        return 0
    # Checkmate: No legal moves and king is in check
    elif legal_moves == 0 and in_check:
        return -MATE_SCORE - depth

    if -MATE_SCORE < best_score < MATE_SCORE:
        record_tt_entry(engine, position, best_score, tt_hash_flag, best_move, depth)

    # We return our best score possible. This is an 'All node' and we have failed low
    return best_score


# An iterative search approach to negamax
# @nb.njit(nb.void(Search.class_type.instance_type, Position.class_type.instance_type, nb.boolean), cache=True)
# @nb.njit(cache=False)
def iterative_search(engine, position, compiling):

    # engine.start_time = get_time()
    SearchStruct_set_start_time(engine, get_time())

    # Reset engine variables
    reset_search(engine)

    original_side = position.side

    # Prepare window for negamax search
    alpha = -INF
    beta = INF

    running_depth = 1

    best_pv = ["" for _ in range(0)]
    best_score = 0

    average_branching_factor = 1
    prev_node_count = 1
    full_searches = 0  # These are the number of searches that aren't returned from TT immediately.

    while running_depth <= engine.max_depth:

        # Reset engine variables
        # engine.current_search_depth = running_depth
        SearchStruct_set_current_search_depth(engine, running_depth)

        # Negamax search
        returned = negamax(engine, position, alpha, beta, running_depth, False)

        # Reset the window
        if returned <= alpha or returned >= beta:
            alpha = -INF
            beta = INF
            continue

        # Adjust aspiration window
        alpha = returned - ASPIRATION_VAL
        beta = returned + ASPIRATION_VAL

        # Obtain principle variation line and print it
        pv_line = []
        for c in range(engine.pv_length[0]):
            pv_line.append(get_uci_from_move(engine.pv_table[0][c]))
            # position.side ^= 1
            PositionStruct_set_side(position, position.side ^ 1)

        # position.side = original_side
        PositionStruct_set_side(position, original_side)

        best_pv = pv_line if not engine.stopped and len(pv_line) else best_pv
        best_score = returned if not engine.stopped else best_score

        lapsed_time = get_time() - engine.start_time

        if engine.stopped:
            running_depth -= 1

        if not compiling:
            print("info depth", running_depth, "score cp", best_score,
                  "time", int(lapsed_time * 1000), "nodes", engine.node_count,
                  "nps", int(engine.node_count / max(lapsed_time, 0.0001)), "pv", ' '.join(best_pv))

        if engine.stopped:
            break

        if best_score >= MATE_SCORE:
            break

        if engine.node_count != running_depth and running_depth > 1:

            if full_searches >= 1:
                average_branching_factor *= full_searches
                average_branching_factor += engine.node_count / prev_node_count * 3
                average_branching_factor /= full_searches + 3

                uncertainty = ((running_depth / (running_depth + 3)) + (full_searches / (full_searches + 2))) / 2

                if average_branching_factor * uncertainty * lapsed_time * 1000 > engine.max_time:
                    break

            full_searches += 1

        prev_node_count = engine.node_count
        running_depth += 1

    if not compiling:
        print("bestmove", best_pv[0])

    return


def compile_engine(engine, position):
    start_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    parse_fen(position, start_fen)

    SearchStruct_set_max_depth(engine, 2)
    SearchStruct_set_max_time(engine, 20)
    iterative_search(engine, position, True)
    SearchStruct_set_max_time(engine, 10)
    SearchStruct_set_max_depth(engine, 30)

    return


''' No ordering
1 0.1986857079999993 0.58 1 1 5 5
2 0.19898225000000025 0.0 21 22 94 99
3 0.20294883300000066 0.58 277 299 796 895
4 0.2540480830000007 0.0 4213 4512 15121 16016
5 0.7658198330000001 0.41 39801 44313 123560 139576
6 4.629144374999999 0.0 296778 341091 1317306 1456882
7 43.06958825 0.3 2639583 2980674 11574655 13031537
UNFINISHED 60.017578625 1144000 4124674 5518826 18550363

Move Ordering!!
1 0.2423271249999992 97 76 0.58 1 1 1 1
2 0.24245779199999973 97 76 0.0 21 22 20 21
3 0.24329862499999955 97 76 0.58 60 82 20 41
4 0.2465868750000002 97 76 0.0 524 606 556 597
5 0.2638864590000001 97 76 0.41 1424 2030 910 1507
6 0.38953879199999975 97 76 0.0 14077 16107 30927 32434
7 1.2136407499999997 85 65 0.3 58633 74740 109453 141887
8 8.362156709 97 76 0.0 445415 520155 1999461 2141348
UNFINISHED 60.012888958999994 1906000 2426155 13584540 15725888

PV sorting!
1 0.1396929999999994 0.44 1 1 1 1
b1c3
2 0.1397954170000002 0.0 21 22 20 21
b1c3 g8f6
3 0.1404795829999994 0.44 60 82 20 41
b1c3 g8f6 g1f3
4 0.14281887499999968 0.0 524 606 576 617
b1c3 g8f6 g1f3 b8c6
5 0.1589633750000008 0.42 1907 2513 1806 2423
g1f3 g8f6 d2d4 b8c6 b1d2
6 0.2554438749999992 0.0 15012 17525 35136 37559
g1f3 g8f6 d2d4 d7d5 b1d2 b8d7
7 1.1604902080000006 0.27 76131 93656 216360 253919
e2e4 b8c6 g1e2 g8f6 d2d3 e7e5 b1d2
8 9.617176167 0.0 559200 652856 3375991 3629910
e2e4 b8c6 g1f3 g8f6 b1c3 e7e5 f1b5 d7d6
8 60.001914041999996 0.0 2270208 2923064 18240286 21870196
e2e4 b8c6 d2d4 d7d5 e4d5 d8d5 g1e2 g8f6 b1c3

Principle Variation Search!!
1 0.7536724580000005 0.44 1 1 1 1
b1c3
2 0.7537917919999995 0.0 21 22 20 21
b1c3 g8f6
3 0.7544873750000001 0.44 60 82 20 41
b1c3 g8f6 g1f3
4 0.7568602080000009 0.0 524 606 571 612
b1c3 g8f6 g1f3 b8c6
5 0.7750874999999997 0.42 2401 3007 2637 3249
g1f3 g8f6 d2d4 b8c6 b1d2
6 0.8589477920000004 0.0 13334 16341 29039 32288
g1f3 g8f6 d2d4 d7d5 b1d2 b8d7
7 2.0724501249999996 0.27 112205 128546 341214 373502
e2e4 b8c6 g1e2 g8f6 d2d3 e7e5 b1d2
8 7.887618207999999 0.0 448857 577403 2237623 2611125
e2e4 b8c6 g1f3 g8f6 b1c3 e7e5 f1b5 d7d6
9 46.730271417 0.38 2193773 2771176 11069776 13680901
e2e4 b8c6 d2d4 d7d5 e4d5 d8d5 g1e2 g8f6 b1c3
9 60.000528792 0.0 559104 3330280 5145567 18826468

Killer and History moves!
1 0.42398825000000073 0.44 1 1 1 1
b1c3
2 0.4240921669999995 0.0 21 22 20 21
b1c3 g8f6
3 0.4249252089999995 0.44 60 82 32 53
b1c3 g8f6 g1f3
4 0.42729479199999965 0.0 525 607 597 650
b1c3 g8f6 g1f3 b8c6
5 0.4462890420000001 0.42 2467 3074 3043 3693
g1f3 g8f6 d2d4 b8c6 b1d2
6 0.5262448339999999 0.0 13374 16448 26674 30367
g1f3 g8f6 d2d4 d7d5 b1d2 b8d7
7 1.4161948340000006 0.27 104073 120521 164775 195142
e2e4 b8c6 g1e2 g8f6 d2d3 e7e5 b1d2
8 4.604588 0.0 395346 515867 824946 1020088
e2e4 b8c6 g1f3 g8f6 b1c3 d7d5 e4e5 f6d7
9 32.498212167 0.38 2489259 3005126 5737465 6757553
e2e4 b8c6 d2d4 d7d5 e4d5 d8d5 g1e2 g8f6 b1c3
9 60.032104167 0.0 1997824 5002950 7200725 13958278

Late move reductions!!!
1 0.28811379200000076 0.44 1 1 2 2
b1c3
2 0.2882701670000003 0.0 22 23 21 23
b1c3 g8f6
3 0.28888645800000035 0.44 65 88 70 93
b1c3 g8f6 g1f3
4 0.29124329200000076 0.0 546 634 632 725
b1c3 g8f6 g1f3 b8c6
5 0.3020930830000008 0.42 1897 2531 3031 3756
g1f3 b8c6 d2d4 g8f6 b1d2
6 0.35967512500000076 0.0 9735 12266 18299 22055
g1f3 g8f6 d2d4 d7d5 b1d2 b8d7
7 0.4427393750000004 0.24 12432 24698 22708 44763
g1f3 d7d5 d2d4 g8f6 b1c3 b8d7 e2e3
8 1.0323145829999998 0.03 78622 103320 171652 216415
e2e4 b8c6 d2d4 e7e5 d4d5 c6e7 d5d6 c7d6
9 5.0681207499999985 0.24 434329 537649 1183299 1399714
d2d4 d7d5 b1c3 g8f6 g1f3 e7e6 e2e3 b8d7 f1b5
10 13.018592583 0.3 803640 1341289 2229399 3629113
e2e4 e7e5 b1c3 g8f6 g1f3 b8c6 f1b5 d7d6 b5c6 b7c6 e1g1
11 58.954614250000006 0.22 3606357 4947646 15262090 18891203
g1f3 g8f6 d2d4 d7d5 e2e3 c8e6 b1c3 b8d7 f1b5 a7a6 b5d3
11 60.000863417000005 0.0 138240 5085886 101753 18992956
b1c3

Null move pruning!!! -- probably has even more effect in tactical positions
1 0.35565758399999936 0.44 1 1 2 2
b1c3
2 0.3558044169999999 0.0 22 23 21 23
b1c3 g8f6
3 0.3564155840000005 0.44 65 88 70 93
b1c3 g8f6 g1f3
4 0.35870662499999995 0.0 546 634 632 725
b1c3 g8f6 g1f3 b8c6
5 0.3676698750000007 0.42 1417 2051 2386 3111
g1f3 b8c6 d2d4 g8f6 b1d2
6 0.4175918749999994 0.0 8566 10617 16208 19319
g1f3 g8f6 d2d4 d7d5 b1d2 b8d7
7 0.5121429170000003 0.25 13544 24161 26782 46101
d2d4 g8f6 g1f3 d7d5 e2e3 b8d7 b1d2
8 0.8462289169999995 0.17 42629 66790 105024 151125
d2d4 g8f6 b1d2 b8c6 g1f3 d7d6 e2e4 e7e5
9 1.4512001249999997 0.24 63785 130575 198323 349448
d2d4 g8f6 b1c3 d7d5 g1f3 e7e6 e2e3 b8d7 f1b5
10 5.951150459000001 0.17 414777 545352 1376480 1725928
d2d4 g8f6 b1c3 b8c6 e2e4 d7d5 e4e5 f6e4 g1e2 e7e6
11 46.158172834 0.17 2742684 3288036 13305334 15031262
g1f3 g8f6 d2d4 d7d5 b1d2 e7e6 e2e3 b8d7 f1b5 a7a6 b5d3
11 60.017434709 0.0 824320 4112356 4571589 19602851
b1c3 e7e5

Aspiration Windows !! -- Also included Q nodes in node count since that is the standard
info depth 1 score cp 44 time 0 nodes 3 nps 10 pv b1c3
info depth 2 score cp 0 time 0 nodes 43 nps 161 pv b1c3 g8f6
info depth 3 score cp 44 time 0 nodes 127 nps 607 pv b1c3 g8f6 g1f3
info depth 4 score cp 0 time 0 nodes 1190 nps 4728 pv b1c3 g8f6 g1f3 b8c6
info depth 5 score cp 42 time 0 nodes 3821 nps 17523 pv g1f3 b8c6 d2d4 g8f6 b1d2
info depth 6 score cp 0 time 0 nodes 25307 nps 87949 pv g1f3 g8f6 d2d4 d7d5 b1d2 b8d7
info depth 7 score cp 30 time 0 nodes 38678 nps 157634 pv d2d4 g8f6 g1f3 d7d5 e2e3 b8d7 b1d2
info depth 8 score cp 12 time 0 nodes 159573 nps 283935 pv g1f3 b8c6 d2d4 d7d6 b1d2 g8f6 e2e4 e7e5
info depth 9 score cp 24 time 1 nodes 422541 nps 359187 pv g1f3 g8f6 e2e3 d7d5 b1c3 e7e6 d2d4 b8d7 f1b5
info depth 10 score cp 23 time 12 nodes 4363782 nps 399351 pv e2e4 e7e5 b1c3 d7d6 g1f3 g8e7 f1c4 c8g4 h2h3 g4e6
info depth 11 score cp 16 time 59 nodes 18906158 nps 399146 pv d2d4 d7d5 g1f3 e7e6 e2e3 b8c6 b1d2 g8f6 f1b5 a7a6 b5d3
info depth 11 score cp 0 time 60 nodes 30721 nps 399191 pv b1c3
bestmove d2d4

History moves fixed to be HH += depth * depth instead of HH += depth
info depth 1 score cp 44 time 883 nodes 11 nps 12 pv b1c3
info depth 2 score cp 0 time 883 nodes 51 nps 70 pv b1c3 g8f6
info depth 3 score cp 44 time 884 nodes 141 nps 229 pv b1c3 g8f6 g1f3
info depth 4 score cp 0 time 887 nodes 1182 nps 1560 pv b1c3 g8f6 g1f3 b8c6
info depth 5 score cp 42 time 895 nodes 3701 nps 5681 pv g1f3 b8c6 d2d4 g8f6 b1d2
info depth 6 score cp 0 time 946 nodes 25343 nps 32146 pv g1f3 g8f6 d2d4 d7d5 b1d2 b8d7
info depth 7 score cp 30 time 1038 nodes 38616 nps 66465 pv d2d4 g8f6 g1f3 d7d5 e2e3 b8d7 b1d2
info depth 8 score cp 1 time 1730 nodes 312448 nps 220449 pv d2d4 g8f6 g1f3 d7d5 e2e3 b8c6 b1d2 e7e6
info depth 9 score cp 25 time 2266 nodes 210289 nps 261143 pv b1c3 g8f6 e2e4 b8c6 d2d3 d7d5 g1f3 e7e6 c1g5
info depth 10 score cp 27 time 7272 nodes 1974641 nps 352908 pv e2e4 b8c6 d2d4 d7d5 e4d5 d8d5 g1e2 e7e6 b1c3 f8b4
info depth 11 score cp 15 time 52360 nodes 17602321 nps 385188 pv g1f3 g8f6 d2d4 d7d5 b1d2 e7e6 e2e3 b8d7 f1b5 a7a6 b5d3
info depth 11 score cp 15 time 60001 nodes 2907338 nps 384593 pv g1f3 g8f6 d2d4 d7d5 b1d2 e7e6 e2e3 b8d7 f1b5 a7a6 b5d3
bestmove g1f3


Quiescence search terminal nodes counted since this is the default
    PST also hand-tuned to somewhat match PESTO eval
info depth 1 score cp 94 time 62 nodes 26 nps 418 pv b1c3
info depth 2 score cp 50 time 62 nodes 86 nps 1801 pv b1c3 g8f6
info depth 3 score cp 94 time 62 nodes 211 nps 5171 pv b1c3 g8f6 g1f3
info depth 4 score cp 50 time 64 nodes 1564 nps 29099 pv b1c3 g8f6 g1f3 b8c6
info depth 5 score cp 91 time 70 nodes 3196 nps 72436 pv b1c3 b8c6 g1f3 g8f6 d2d4
info depth 6 score cp 51 time 116 nodes 34879 nps 341784 pv g1f3 b8c6 d2d4 g8f6 b1d2 d7d5
info depth 7 score cp 79 time 152 nodes 27991 nps 445387 pv g1f3 g8f6 b1c3 d7d5 d2d4 b8d7 e2e3
info depth 8 score cp 60 time 500 nodes 261622 nps 658032 pv b1c3 g8f6 e2e4 d7d5 f1d3 d5e4 c3e4 b8d7
info depth 9 score cp 54 time 970 nodes 360442 nps 710736 pv b1c3 g8f6 e2e4 b8c6 d2d4 d7d5 e4d5 f6d5 g1e2
info depth 10 score cp 64 time 2026 nodes 802543 nps 736425 pv g1f3 g8f6 b1c3 e7e6 d2d4 b8c6 c1f4 d7d5 f3e5 f6e4
info depth 11 score cp 64 time 11688 nodes 7311427 nps 753246 pv b1c3 e7e6 g1f3 d7d5 d2d4 g8f6 e2e3 b8d7 f1b5 a7a6 b5d3
info depth 11 score cp 64 time 60000 nodes 37037138 nps 764014 pv b1c3 e7e6 g1f3 d7d5 d2d4 g8f6 e2e3 b8d7 f1b5 a7a6 b5d3
bestmove b1c3


LMR full depth moves changed to 3
info depth 1 score cp 94 time 634 nodes 26 nps 41 pv b1c3
info depth 2 score cp 50 time 634 nodes 86 nps 176 pv b1c3 g8f6
info depth 3 score cp 94 time 634 nodes 211 nps 509 pv b1c3 g8f6 g1f3
info depth 4 score cp 50 time 638 nodes 1564 nps 2957 pv b1c3 g8f6 g1f3 b8c6
info depth 5 score cp 91 time 642 nodes 3196 nps 7915 pv b1c3 b8c6 g1f3 g8f6 d2d4
info depth 6 score cp 51 time 688 nodes 34879 nps 58035 pv g1f3 b8c6 d2d4 g8f6 b1d2 d7d5
info depth 7 score cp 79 time 723 nodes 27991 nps 93859 pv g1f3 g8f6 b1c3 d7d5 d2d4 b8d7 e2e3
info depth 8 score cp 60 time 1064 nodes 261622 nps 309488 pv b1c3 g8f6 e2e4 d7d5 f1d3 d5e4 c3e4 b8d7
info depth 9 score cp 54 time 1527 nodes 360442 nps 451651 pv b1c3 g8f6 e2e4 b8c6 d2d4 d7d5 e4d5 f6d5 g1e2
info depth 10 score cp 64 time 2566 nodes 802543 nps 581482 pv g1f3 g8f6 b1c3 e7e6 d2d4 b8c6 c1f4 d7d5 f3e5 f6e4
info depth 11 score cp 64 time 12071 nodes 7311427 nps 729341 pv b1c3 e7e6 g1f3 d7d5 d2d4 g8f6 e2e3 b8d7 f1b5 a7a6 b5d3
info depth 12 score cp 65 time 59882 nodes 37513705 nps 773475 pv e2e4 e7e6 g1f3 b8c6 b1c3 g8f6 d2d3 d7d5 e4e5 f6d7 
    c1g5 f8e7
info depth 12 score cp 65 time 60001 nodes 1 nps 771947 pv e2e4 e7e6 g1f3 b8c6 b1c3 g8f6 d2d3 d7d5 e4e5 f6d7 c1g5 f8e7
bestmove e2e4

Incremental hash updates
info depth 1 score cp 94 time 264 nodes 26 nps 98 pv b1c3
info depth 2 score cp 50 time 264 nodes 86 nps 423 pv g8f6 b1c3
info depth 3 score cp 94 time 264 nodes 211 nps 1219 pv g8f6 b1c3 b8c6
info depth 4 score cp 50 time 268 nodes 1564 nps 7031 pv b1c3 g8f6 g1f3 b8c6
info depth 5 score cp 91 time 272 nodes 3196 nps 18662 pv b1c3 b8c6 g1f3 g8f6 d2d4
info depth 6 score cp 51 time 317 nodes 34879 nps 125855 pv b8c6 g1f3 e7e5 b1c3 g8e7 e2e4
info depth 7 score cp 79 time 352 nodes 27991 nps 192764 pv b8c6 b1c3 g8f6 e2e4 e7e5 g1e2 d7d6
info depth 8 score cp 60 time 690 nodes 261622 nps 477411 pv b1c3 g8f6 e2e4 d7d5 f1d3 d5e4 c3e4 b8d7
info depth 9 score cp 54 time 1148 nodes 360442 nps 600652 pv b1c3 g8f6 e2e4 b8c6 d2d4 d7d5 e4d5 f6d5 g1e2
info depth 10 score cp 64 time 2165 nodes 802543 nps 689136 pv b8c6 b1c3 g8f6 d2d3 e7e5 g1f3 f8c5 e2e4 c6d4 c3d5
info depth 11 score cp 64 time 11508 nodes 7311427 nps 764972 pv g8f6 d2d3 b8c6 e2e4 e7e5 b1c3 d7d6 g1e2 c8g4 h2h3 g4e6
info depth 12 score cp 65 time 57681 nodes 37513705 nps 802994 pv e2e4 e7e6 g1f3 b8c6 b1c3 g8f6 d2d3 d7d5 e4e5 f6d7 
    c1g5 f8e7
info depth 12 score cp 65 time 60000 nodes 1 nps 771953 pv e2e4 e7e6 g1f3 b8c6 b1c3 g8f6 d2d3 d7d5 e4e5 f6d7 c1g5 f8e7
bestmove e2e4

Eval made efficient
info depth 1 score cp 94 time 90 nodes 26 nps 285 pv b1c3
info depth 2 score cp 50 time 91 nodes 86 nps 1230 pv b1c3 g8f6
info depth 3 score cp 94 time 91 nodes 211 nps 3535 pv b1c3 g8f6 g1f3
info depth 4 score cp 50 time 95 nodes 1564 nps 19810 pv b1c3 g8f6 g1f3 b8c6
info depth 5 score cp 91 time 99 nodes 3196 nps 51146 pv b1c3 b8c6 g1f3 g8f6 d2d4
info depth 6 score cp 51 time 144 nodes 34879 nps 276041 pv g1f3 b8c6 d2d4 g8f6 b1d2 d7d5
info depth 7 score cp 79 time 178 nodes 27991 nps 379822 pv g1f3 g8f6 b1c3 d7d5 d2d4 b8d7 e2e3
info depth 8 score cp 60 time 518 nodes 261622 nps 635433 pv b1c3 g8f6 e2e4 d7d5 f1d3 d5e4 c3e4 b8d7
info depth 9 score cp 54 time 969 nodes 360442 nps 711948 pv b1c3 g8f6 e2e4 b8c6 d2d4 d7d5 e4d5 f6d5 g1e2
info depth 10 score cp 64 time 1976 nodes 802543 nps 755082 pv g1f3 g8f6 b1c3 e7e6 d2d4 b8c6 c1f4 d7d5 f3e5 f6e4
info depth 11 score cp 64 time 11124 nodes 7311427 nps 791401 pv b1c3 e7e6 g1f3 d7d5 d2d4 g8f6 e2e3 b8d7 f1b5 a7a6 b5d3
info depth 12 score cp 65 time 55872 nodes 37513705 nps 828994 pv e2e4 e7e6 g1f3 b8c6 b1c3 g8f6 d2d3 d7d5 e4e5 f6d7 
c1g5 f8e7

info depth 12 score cp 65 time 60000 nodes 1 nps 771958 pv e2e4 e7e6 g1f3 b8c6 b1c3 g8f6 d2d3 d7d5 e4e5 f6d7 c1g5 f8e7
bestmove e2e4

info depth 12 score cp 65 time 1200001 nodes 1048079573 nps 911996 pv e2e4 e7e6 g1f3 b8c6 b1c3 g8f6 d2d3 d7d5 e4e5 f6d7 
    c1g5 f8e7
    
bestmove e2e4


MANY BUG FIXES. 
Move scorer wasn't scoring some quiet moves.
Search went straight to LMR/PVS instead of searching first moves.
info depth 1 score cp 94 time 841 nodes 21 nps 24 pv b1c3
info depth 2 score cp 50 time 841 nodes 60 nps 96 pv b1c3 g8f6
info depth 3 score cp 94 time 842 nodes 159 nps 285 pv b1c3 g8f6 g1f3
info depth 4 score cp 50 time 844 nodes 1437 nps 1986 pv b1c3 g8f6 g1f3 b8c6
info depth 5 score cp 91 time 847 nodes 1649 nps 3922 pv b1c3 g8f6 g1f3 b8c6 e2e4
info depth 6 score cp 51 time 888 nodes 30906 nps 38510 pv b1c3 g8f6 e2e4 b8c6 g1e2 d7d5
info depth 7 score cp 80 time 966 nodes 62186 nps 99720 pv g1f3 g8f6 d2d4 b8c6 b1c3 d7d5 e2e3
info depth 8 score cp 51 time 1305 nodes 263405 nps 275541 pv g1f3 g8f6 d2d4 d7d5 b1c3 b8c6 f3e5 c6e5
info depth 9 score cp 54 time 1918 nodes 494649 nps 445374 pv b1c3 g8f6 e2e4 b8c6 d2d4 d7d5 e4d5 f6d5 g1e2
info depth 10 score cp 74 time 6182 nodes 3415069 nps 690630 pv e2e4 e7e6 b1c3 d7d5 g1f3 g8f6 f1d3 d5e4 c3e4 b8d7
info depth 11 score cp 64 time 19787 nodes 11045537 nps 773996 pv b1c3 g8f6 e2e4 e7e5 g1f3 f8b4 f3e5 d7d6 e5d3 b4c3 d2c3
info depth 11 score cp 64 time 60000 nodes 34169971 nps 824745 pv b1c3 g8f6 e2e4 e7e5 g1f3 f8b4 f3e5 d7d6 e5d3 b4c3 d2c3
bestmove b1c3


Transposition Table returns
info depth 1 score cp 94 time 719 nodes 21 nps 29 pv b1c3
info depth 2 score cp 50 time 719 nodes 60 nps 112 pv b1c3 g8f6
info depth 3 score cp 94 time 719 nodes 240 nps 445 pv g1f3 b8c6 b1c3
info depth 4 score cp 50 time 723 nodes 1395 nps 2371 pv g1f3 b8c6 b1c3
info depth 5 score cp 50 time 726 nodes 1701 nps 4705 pv g1f3 g8f6 b1c3
info depth 6 score cp 50 time 762 nodes 25227 nps 37548 pv g1f3 b8c6 b1c3 d7d5 e2e4 g8f6
info depth 7 score cp 57 time 814 nodes 38788 nps 82815 pv b1c3 g8f6 e2e4 d7d5 e4d5 f6d5 g1e2
info depth 8 score cp 57 time 1153 nodes 255182 nps 279702 pv e2e4 g8f6 e4e5 f6d5 c2c4 d5b4 a2a3 b4c6
info depth 9 score cp 74 time 1658 nodes 390389 nps 429823 pv b1c3 d7d5 g1f3 g8f6 d2d4 e7e6 e2e3 b8d7 f1b5
info depth 10 score cp 64 time 3179 nodes 1170370 nps 592353 pv g1f3 g8f6 b1c3 d7d5 d2d4 e7e6 f3e5 b8c6 c1f4 f6e4
info depth 11 score cp 72 time 16518 nodes 10231402 nps 733385 pv e2e4 e7e6 b1c3 f8b4 g1e2 g8e7 a2a3 b4c3 d2c3 e8g8 e2d4
info depth 11 score cp 72 time 60001 nodes 35675258 nps 796482 pv e2e4 e7e6 b1c3 f8b4 g1e2 g8e7 a2a3 b4c3 d2c3 e8g8 e2d4
bestmove e2e4

TT move ordering
info depth 1 score cp 94 time 816 nodes 21 nps 25 pv b1c3
info depth 2 score cp 50 time 816 nodes 60 nps 99 pv b1c3 g8f6
info depth 3 score cp 94 time 816 nodes 240 nps 393 pv g1f3 b8c6 b1c3
info depth 4 score cp 50 time 821 nodes 1395 nps 2088 pv g1f3 b8c6 b1c3
info depth 5 score cp 50 time 824 nodes 1701 nps 4143 pv g1f3 g8f6 b1c3
info depth 6 score cp 50 time 862 nodes 25214 nps 33184 pv g1f3 b8c6 b1c3 d7d5 e2e4 g8f6
info depth 7 score cp 57 time 919 nodes 41435 nps 76218 pv b1c3 g8f6 e2e4 d7d5 e4d5 f6d5 g1e2
info depth 8 score cp 51 time 1177 nodes 152137 nps 188748 pv b1c3 g8f6 d2d4 b8c6 g1f3 d7d5 f3e5 c6e5
info depth 9 score cp 65 time 1908 nodes 510287 nps 383840 pv d2d4 d7d5 b1c3 g8f6 g1f3 e7e6 f3e5 b8c6 c1f4
info depth 10 score cp 64 time 3003 nodes 769888 nps 500278 pv b1c3 b8c6 d2d4 d7d5 g1f3 g8f6 c1f4 e7e6 f3e5 f6e4
info depth 11 score cp 69 time 9143 nodes 4556303 nps 662629 pv e2e4 e7e6 b1c3 b8c6 d2d4 d7d5 g1e2 g8f6 c1g5 f8b4 e2g3
info depth 11 score cp 69 time 60000 nodes 38790260 nps 747477 pv e2e4 e7e6 b1c3 b8c6 d2d4 d7d5 g1e2 g8f6 c1g5 f8b4 e2g3
bestmove e2e4

bug fixes
info depth 1 score cp 44 time 123 nodes 21 nps 169 pv b1c3
info depth 2 score cp 0 time 124 nodes 60 nps 652 pv b1c3 g8f6
info depth 3 score cp 44 time 124 nodes 240 nps 2576 pv g1f3 b8c6 b1c3
info depth 4 score cp 0 time 128 nodes 1316 nps 12734 pv g1f3 b8c6 b1c3
info depth 5 score cp 41 time 142 nodes 3785 nps 38159 pv d2d4 g8f6 g1f3 f6e4 f3e5
info depth 6 score cp 41 time 149 nodes 4791 nps 68195 pv d2d4 b8c6 g1f3 g8f6
info depth 7 score cp 29 time 202 nodes 33884 nps 218007 pv d2d4 d7d5 b1c3 g8f6 g1f3 b8d7 e2e3
info depth 8 score cp 12 time 501 nodes 206689 nps 499958 pv b1c3 b8c6 g1f3 g8f6 d2d4 e7e6 f3e5 c6e5
info depth 9 score cp 24 time 866 nodes 266991 nps 597815 pv d2d4 e7e6 b1c3 g8f6 g1f3 d7d5 e2e3 b8d7 f1b5
info depth 10 score cp 13 time 3473 nodes 1961818 nps 713761 pv d2d4 d7d5 b1c3 g8f6 g1f3 b8c6 f3e5 c6e5 d4e5 f6g4
info depth 11 score cp 23 time 6970 nodes 2660558 nps 737453 pv b1c3 e7e6 g1f3 b8c6 e2e4 d7d5 d2d3 d5e4 c3e4 f8b4 
c1d2 g8f6
info depth 11 score cp 23 time 60001 nodes 43813984 nps 815881 pv b1c3 e7e6 g1f3 b8c6 e2e4 d7d5 d2d3 d5e4 c3e4 f8b4 
c1d2 g8f6
bestmove b1c3


HUGE CHANGES. flip_position() removed, move_gen and all make_move undo_move functions changed.
Incremental hashing is fixed.
info depth 1 score cp 44 time 754 nodes 21 nps 27 pv b1c3
info depth 2 score cp 0 time 755 nodes 60 nps 107 pv b1c3 b8c6
info depth 3 score cp 44 time 755 nodes 163 nps 322 pv b1c3 g8f6 g1f3
info depth 4 score cp 0 time 757 nodes 1343 nps 2094 pv b1c3 g8f6 g1f3 b8c6
info depth 5 score cp 42 time 761 nodes 2998 nps 6024 pv d2d4 b8c6 g1f3 g8f6 b1d2
info depth 6 score cp 1 time 776 nodes 12845 nps 22460 pv g1f3 g8f6 d2d4 b8c6
info depth 7 score cp 30 time 789 nodes 15226 nps 41373 pv g1f3 g8f6 d2d4 d7d5 b1d2 b8d7 e2e3
info depth 8 score cp 1 time 879 nodes 90656 nps 140242 pv e2e3 b8c6 g1f3 g8f6
info depth 9 score cp 25 time 1127 nodes 260358 nps 340410 pv g1f3 d7d5 d2d4 g8f6 b1d2 b8d7 e2e3
info depth 10 score cp 14 time 1431 nodes 315590 nps 488377 pv g1f3 d7d5 d2d4 g8f6 b1c3 b8d7 c1f4 e7e6 f3e5 f6e4
info depth 11 score cp 30 time 1923 nodes 542702 nps 645741 pv g1f3 d7d5 d2d4 b8d7 e2e3 g8f6 b1d2 a7a6 f3e5 d7e5 d4e5
info depth 12 score cp 34 time 7538 nodes 6043636 nps 966415 pv g1f3 d7d5 d2d4 b8d7 e2e3 g8f6 b1d2 f6e4 f1b5 e4d2
 c1d2 e7e6
 
info depth 13 score cp 21 time 44051 nodes 39736081 nps 1067423 pv d2d4 e7e6 e2e4 b8c6 b1c3 g8f6 g1e2 d7d5 e4d5
info depth 13 score cp 0 time 60000 nodes 17928359 nps 1082486 pv d2d4 e7e6 e2e4 b8c6 b1c3 g8f6 g1e2 d7d5 e4d5
bestmove d2d4

tt move scoring changed from 30000 to 100000; pv scoring removed
info depth 1 score cp 44 time 115 nodes 21 nps 181 pv b1c3
info depth 2 score cp 0 time 116 nodes 60 nps 697 pv b1c3 b8c6
info depth 3 score cp 44 time 116 nodes 163 nps 2098 pv b1c3 g8f6 g1f3
info depth 4 score cp 0 time 118 nodes 1343 nps 13349 pv b1c3 g8f6 g1f3 b8c6
info depth 5 score cp 42 time 122 nodes 2998 nps 37386 pv d2d4 b8c6 g1f3 g8f6 b1d2
info depth 6 score cp 1 time 169 nodes 12845 nps 103134 pv g1f3 g8f6 d2d4 b8c6
info depth 7 score cp 30 time 197 nodes 15225 nps 165195 pv g1f3 g8f6 d2d4 d7d5 b1d2 b8d7 e2e3
info depth 8 score cp 1 time 288 nodes 90656 nps 428153 pv e2e3 b8c6 g1f3 g8f6
info depth 9 score cp 25 time 530 nodes 260466 nps 724069 pv g1f3 d7d5 d2d4 g8f6 b1d2 b8d7 e2e3
info depth 10 score cp 14 time 819 nodes 315666 nps 853777 pv g1f3 d7d5 d2d4 g8f6 b1c3 b8d7 c1f4 e7e6 f3e5 f6e4
info depth 11 score cp 30 time 1287 nodes 542463 nps 964635 pv g1f3 d7d5 d2d4 b8d7 e2e3 g8f6 b1d2 a7a6 f3e5 d7e5 d4e5
info depth 12 score cp 27 time 3299 nodes 2310499 nps 1076531 pv g1f3 d7d5 d2d4 b8d7 e2e3 g8f6 f1d3 e7e6 e1g1 f8b4
 c1d2 b4d6
info depth 13 score cp 26 time 19586 nodes 18783385 nps 1140394 pv e2e4 b8c6 g1f3 e7e6 b1c3 g8f6 e4e5 f6d5 c3d5 e6d5
 d2d4 f8b4 c2c3 b4e7
info depth 13 score cp 12 time 60000 nodes 47812719 nps 1169127 pv e2e4 b8c6 g1f3 e7e6 b1c3 g8f6 e4e5 f6d5 c3d5 e6d5
 d2d4 f8b4 c2c3 b4e7
bestmove e2e4


Mating scores not stored in tt
info depth 1 score cp 44 time 587 nodes 21 nps 35 pv b1c3
info depth 2 score cp 0 time 587 nodes 60 nps 137 pv b1c3 b8c6
info depth 3 score cp 44 time 587 nodes 163 nps 415 pv b1c3 g8f6 g1f3
info depth 4 score cp 0 time 589 nodes 1343 nps 2691 pv b1c3 g8f6 g1f3 b8c6
info depth 5 score cp 42 time 592 nodes 2998 nps 7734 pv d2d4 b8c6 g1f3 g8f6 b1d2
info depth 6 score cp 1 time 606 nodes 12845 nps 28746 pv g1f3 g8f6 d2d4 b8c6
info depth 7 score cp 30 time 619 nodes 15225 nps 52715 pv g1f3 g8f6 d2d4 d7d5 b1d2 b8d7 e2e3
info depth 8 score cp 1 time 706 nodes 90660 nps 174636 pv e2e3 b8c6 g1f3 g8f6
info depth 9 score cp 25 time 940 nodes 260707 nps 408433 pv g1f3 d7d5 d2d4 g8f6 b1d2 b8d7 e2e3
info depth 10 score cp 14 time 1225 nodes 314416 nps 569739 pv g1f3 d7d5 d2d4 g8f6 b1c3 b8d7 c1f4 e7e6 f3e5 f6e4
info depth 11 score cp 30 time 1698 nodes 549654 nps 734905 pv g1f3 d7d5 d2d4 b8d7 e2e3 g8f6 b1d2 a7a6 f3e5 d7e5 d4e5
info depth 12 score cp 27 time 3685 nodes 2286679 nps 959119 pv g1f3 d7d5 d2d4 b8d7 e2e3 g8f6 f1d3 e7e6 e1g1 f8b4 c1d2
 b4d6
info depth 13 score cp 26 time 18457 nodes 16994152 nps 1112252 pv e2e4 b8c6 g1f3 e7e6 b1c3 g8f6 e4e5 f6d5
info depth 13 score cp 26 time 60000 nodes 48475267 nps 1150060 pv e2e4 b8c6 g1f3 e7e6 b1c3 g8f6 e4e5 f6d5
bestmove e2e4

tapered eval
info depth 1 score cp 50 time 254 nodes 21 nps 82 pv b1c3
info depth 2 score cp 0 time 255 nodes 41 nps 242 pv b1c3 b8c6
info depth 3 score cp 50 time 255 nodes 115 nps 692 pv b1c3 b8c6 g1f3
info depth 4 score cp 0 time 257 nodes 138 nps 1224 pv b1c3 b8c6 g1f3 g8f6
info depth 5 score cp 41 time 258 nodes 1371 nps 6520 pv b1c3 b8c6 g1f3 g8f6 d2d4
info depth 6 score cp 0 time 267 nodes 9715 nps 42630 pv b1c3 b8c6 g1f3 g8f6 d2d4 d7d5
info depth 7 score cp 30 time 276 nodes 11315 nps 82134 pv b1c3 b8c6 g1f3 g8f6 d2d4 d7d5 e2e3
info depth 8 score cp 0 time 351 nodes 87981 nps 315083 pv b1c3 b8c6 g1f3 g8f6 d2d4 d7d5 e2e3 e7e6
info depth 9 score cp 23 time 550 nodes 230197 nps 618749 pv g1f3 g8f6 e2e3 d7d5 f1b5 b8c6 e1g1 e7e5 b5c6 c8d7 b1c3
info depth 10 score cp 23 time 786 nodes 279069 nps 788212 pv g1f3 g8f6 e2e3 d7d5 f1b5
info depth 11 score cp 30 time 1362 nodes 687606 nps 959780 pv g1f3 g8f6 e2e3 b8c6 f1b5 e7e5 e1g1 f8c5 b1c3 e8g8 d2d4
info depth 12 score cp 36 time 4105 nodes 3321085 nps 1127448 pv g1f3 b8c6 b1c3 e7e5 e2e4 d8h4 f1b5 h4f2 e1f2 g8f6
 d2d4 d7d5 f2g1
info depth 13 score cp 25 time 7151 nodes 3705078 nps 1165331 pv g1f3 b8c6 b1c3 e7e5 e2e4 g8f6 f1c4 d7d5 d2d4 f8c5
 e1g1 e8g8 c1f4
info depth 13 score cp 25 time 60004 nodes 59030719 nps 1122659 pv g1f3 b8c6 b1c3 e7e5 e2e4 g8f6 f1c4 d7d5 d2d4 f8c5
 e1g1 e8g8 c1f4
bestmove g1f3

info depth 1 score cp 96 time 609 nodes 21 nps 34 pv b1c3
info depth 2 score cp 46 time 609 nodes 41 nps 101 pv b1c3 b8c6
info depth 3 score cp 96 time 609 nodes 115 nps 290 pv b1c3 b8c6 g1f3
info depth 4 score cp 46 time 611 nodes 138 nps 514 pv b1c3 b8c6 g1f3 g8f6
info depth 5 score cp 87 time 613 nodes 1479 nps 2925 pv b1c3 b8c6 g1f3 g8f6 d2d4
info depth 6 score cp 46 time 619 nodes 6813 nps 13882 pv b1c3 b8c6 g1f3 g8f6 d2d4 d7d5
info depth 7 score cp 76 time 645 nodes 30334 nps 60342 pv g1f3 b8c6 b1c3
info depth 8 score cp 48 time 743 nodes 107307 nps 196613 pv b1c3 d7d5 e2e4 g8f6 f1d3 d5e4 c3e4 f6e4
info depth 9 score cp 61 time 930 nodes 213277 nps 386483 pv g1f3 b8c6 b1c3 d7d5 d2d4 g8f6
info depth 10 score cp 57 time 1569 nodes 728373 nps 693048 pv e2e4 b8c6 d2d4 e7e6 b1c3 g8f6 g1e2 f8b4 a2a3 b4c3
info depth 11 score cp 63 time 2519 nodes 1080260 nps 860716 pv e2e4 e7e5 b1c3 g8f6 g1f3 b8c6 d2d4 e5d4 f3d4 f8b4 c1g5
info depth 12 score cp 60 time 5593 nodes 3562806 nps 1024645 pv g1f3 b8c6 d2d4 d7d5 e2e3 g8f6 f1d3 f6e4 e1g1 e7e6 b1c3
 f8b4
info depth 13 score cp 55 time 15071 nodes 10992692 nps 1109654 pv g1f3 b8c6 d2d4 d7d5 e2e3 g8f6 f1b5 a7a6 b5c6
info depth 14 score cp 53 time 39111 nodes 27686504 nps 1135487 pv g1f3 b8c6 d2d4 d7d5 e2e3 g8f6 f1e2 f6e4 e1g1 e7e6
 b1d2 f8b4 c2c3 b4d6
info depth 14 score cp 53 time 60000 nodes 1 nps 740166 pv g1f3 b8c6 d2d4 d7d5 e2e3 g8f6 f1e2 f6e4 e1g1 e7e6 b1d2 f8b4
 c2c3 b4d6
bestmove g1f3

Slight change to ordering captures
info depth 1 score cp 96 time 628 nodes 21 nps 33 pv b1c3
info depth 2 score cp 46 time 628 nodes 41 nps 98 pv b1c3 b8c6
info depth 3 score cp 96 time 629 nodes 115 nps 281 pv b1c3 b8c6 g1f3
info depth 4 score cp 46 time 630 nodes 138 nps 499 pv b1c3 b8c6 g1f3 g8f6
info depth 5 score cp 87 time 632 nodes 1475 nps 2830 pv b1c3 b8c6 g1f3 g8f6 d2d4
info depth 6 score cp 46 time 639 nodes 6802 nps 13443 pv b1c3 b8c6 g1f3 g8f6 d2d4 d7d5
info depth 7 score cp 76 time 665 nodes 30323 nps 58504 pv g1f3 b8c6 b1c3
info depth 8 score cp 48 time 763 nodes 107219 nps 191476 pv b1c3 d7d5 e2e4 g8f6 f1d3 d5e4 c3e4 f6e4
info depth 9 score cp 61 time 947 nodes 209337 nps 375240 pv g1f3 b8c6 b1c3 d7d5 d2d4 g8f6
info depth 10 score cp 53 time 1542 nodes 684381 nps 673939 pv g1f3 d7d5 e2e3 g8f6 f1e2 b8c6 e1g1 d5d4 e2b5 c8g4
info depth 11 score cp 58 time 2178 nodes 673273 nps 786550 pv g1f3 d7d5 e2e3 b8c6 d2d4 g8f6 b1c3 f6e4 f1b5 a7a6 c3e4
info depth 12 score cp 60 time 4409 nodes 2615145 nps 981659 pv g1f3 d7d5 e2e3 b8c6 d2d4 g8f6 f1d3 f6e4 e1g1 e7e6 b1c3
 f8b4
info depth 13 score cp 55 time 8892 nodes 5156809 nps 1066589 pv g1f3 d7d5 e2e3 b8c6 d2d4 g8f6 f1b5 a7a6 b5c6
info depth 14 score cp 51 time 26393 nodes 20660358 nps 1142137 pv g1f3 d7d5 e2e3 b8c6 d2d4 g8f6 f1b5 a7a6 b5e2 f6e4
 e1g1 e7e6 b1c3 f8b4
info depth 14 score cp 51 time 60000 nodes 1 nps 502417 pv g1f3 d7d5 e2e3 b8c6 d2d4 g8f6 f1b5 a7a6 b5e2 f6e4 e1g1 e7e6
 b1c3 f8b4
bestmove g1f3

Adaptive NMP
info depth 1 score cp 96 time 650 nodes 21 nps 32 pv b1c3
info depth 2 score cp 46 time 651 nodes 41 nps 95 pv b1c3 b8c6
info depth 3 score cp 96 time 651 nodes 115 nps 271 pv b1c3 b8c6 g1f3
info depth 4 score cp 46 time 653 nodes 138 nps 482 pv b1c3 b8c6 g1f3 g8f6
info depth 5 score cp 87 time 654 nodes 1475 nps 2733 pv b1c3 b8c6 g1f3 g8f6 d2d4
info depth 6 score cp 46 time 661 nodes 6802 nps 12982 pv b1c3 b8c6 g1f3 g8f6 d2d4 d7d5
info depth 7 score cp 76 time 688 nodes 30382 nps 56636 pv g1f3 b8c6 b1c3
info depth 8 score cp 41 time 800 nodes 117975 nps 195962 pv d2d4 b8c6 b1c3 g8f6 c1f4 d7d6 g1f3 e7e5
info depth 9 score cp 61 time 973 nodes 171842 nps 337837 pv d2d4 d7d5 g1f3 b8c6 b1c3 g8f6 e2e3
info depth 10 score cp 51 time 1657 nodes 747594 nps 649458 pv g1f3 d7d5 e2e3 g8f6 b1c3 b8c6 f1b5 a7a6 b5c6 b7c6 e1g1
info depth 11 score cp 60 time 3299 nodes 1875884 nps 894732 pv e2e4 e7e5 g1f3 g8f6 f3e5 b8c6 e5c6 d7c6 b1c3 c8g4 f1e2
info depth 12 score cp 57 time 6248 nodes 3342972 nps 1007432 pv g1f3 g8f6 d2d4 e7e6 b1d2 d7d5 e2e3 b8c6 f1b5 a7a6 b5c6
 b7c6 e1g1
info depth 13 score cp 49 time 14144 nodes 8892540 nps 1073742 pv g1f3 g8f6 d2d4 e7e6 b1d2 d7d5 e2e3 f8d6 f1d3 e8g8 e1g1
 b8c6 d3b5
info depth 14 score cp 53 time 28672 nodes 16600943 nps 1108686 pv g1f3 g8f6 d2d4 e7e6 e2e3 b8c6 f1e2 f8b4 c2c3 b4e7
 e1g1 d7d5 e2b5 e8g8 b1d2
info depth 14 score cp 53 time 60000 nodes 1 nps 529804 pv g1f3 g8f6 d2d4 e7e6 e2e3 b8c6 f1e2 f8b4 c2c3 b4e7 e1g1 d7d5
 e2b5 e8g8 b1d2
bestmove g1f3


Repetition detection!
info depth 1 score cp 96 time 0 nodes 21 nps 752823 pv b1c3
info depth 2 score cp 46 time 0 nodes 41 nps 157033 pv b1c3 b8c6
info depth 3 score cp 96 time 0 nodes 115 nps 330834 pv b1c3 b8c6 g1f3
info depth 4 score cp 46 time 2 nodes 138 nps 138957 pv b1c3 b8c6 g1f3 g8f6
info depth 5 score cp 87 time 3 nodes 1475 nps 480530 pv b1c3 b8c6 g1f3 g8f6 d2d4
info depth 6 score cp 46 time 10 nodes 6802 nps 839799 pv b1c3 b8c6 g1f3 g8f6 d2d4 d7d5
info depth 7 score cp 76 time 34 nodes 30382 nps 1127363 pv g1f3 b8c6 b1c3
info depth 8 score cp 41 time 139 nodes 117983 nps 1124986 pv d2d4 b8c6 b1c3 g8f6 c1f4 d7d6 g1f3 e7e5
info depth 9 score cp 61 time 287 nodes 171842 nps 1144174 pv d2d4 d7d5 g1f3 b8c6 b1c3 g8f6 e2e3
info depth 10 score cp 51 time 918 nodes 747627 nps 1171866 pv g1f3 d7d5 e2e3 g8f6 b1c3 b8c6 f1b5 a7a6 b5c6 b7c6 e1g1
info depth 11 score cp 60 time 2466 nodes 1875960 nps 1196784 pv e2e4 e7e5 g1f3 g8f6 f3e5 b8c6 e5c6 d7c6 b1c3 c8g4 f1e2
info depth 12 score cp 57 time 5239 nodes 3343637 nps 1201540 pv g1f3 g8f6 d2d4 e7e6 b1d2 d7d5 e2e3 b8c6 f1b5 a7a6
 b5c6 b7c6 e1g1
info depth 13 score cp 49 time 12659 nodes 8878026 nps 1198611 pv g1f3 g8f6 d2d4 e7e6 b1d2 d7d5 e2e3 f8d6 f1d3 e8g8
 e1g1 b8c6 d3b5
info depth 14 score cp 53 time 27275 nodes 17774346 nps 1207988 pv g1f3 g8f6 d2d4 e7e6 e2e3 b8c6 f1e2 f8b4 c2c3 b4e7
 e1g1 d7d5 e2b5 e8g8 b1d2
info depth 14 score cp 53 time 60000 nodes 1 nps 549133 pv g1f3 g8f6 d2d4 e7e6 e2e3 b8c6 f1e2 f8b4 c2c3 b4e7 e1g1
 d7d5 e2b5 e8g8 b1d2
bestmove g1f3

micro-optimizations to PST
info depth 1 score cp 96 time 26 nodes 21 nps 799 pv b1c3
info depth 2 score cp 46 time 26 nodes 41 nps 2316 pv b1c3 b8c6
info depth 3 score cp 96 time 26 nodes 117 nps 6645 pv b1c3 b8c6 g1f3
info depth 4 score cp 46 time 28 nodes 140 nps 11009 pv b1c3 b8c6 g1f3 g8f6
info depth 5 score cp 89 time 30 nodes 1487 nps 59044 pv b1c3 b8c6 g1f3 g8f6 e2e4
info depth 6 score cp 46 time 40 nodes 9325 nps 276354 pv b1c3 b8c6 g1f3 g8f6 e2e4 e7e5
info depth 7 score cp 76 time 51 nodes 12424 nps 458188 pv b1c3 b8c6 g1f3 g8f6 d2d4 d7d5 e2e3
info depth 8 score cp 45 time 140 nodes 94160 nps 838783 pv d2d4 b8c6 e2e4 g8f6 e4e5 f6d5 c2c4 d5b4
info depth 9 score cp 72 time 438 nodes 333551 nps 1028812 pv e2e4 b8c6 d2d4 d7d5
info depth 10 score cp 72 time 669 nodes 269603 nps 1077424 pv e2e4 b8c6 g1f3 e7e5 b1c3 g8f6 d2d4 e5d4 f3d4
info depth 11 score cp 63 time 1105 nodes 514544 nps 1117869 pv e2e4 b8c6 g1f3 e7e5 b1c3 g8f6 f1b5 c6d4 e1g1 d4b5 c3b5
info depth 12 score cp 57 time 4099 nodes 3503317 nps 1156004 pv g1f3 b8c6 d2d4 e7e6 e2e4 g8f6 b1c3
info depth 13 score cp 53 time 9688 nodes 6613393 nps 1171768 pv g1f3 b8c6 d2d4 e7e6 e2e4 g8f6 d4d5 e6d5 e4d5 d8e7
 f1e2 c6e5 e1g1 f6e4
info depth 14 score cp 53 time 28809 nodes 22901060 nps 1188960 pv g1f3 b8c6 d2d4 e7e6 e2e4 g8f6 d4d5 e6d5 e4d5 d8e7
info depth 14 score cp 53 time 60000 nodes 1 nps 570880 pv g1f3 b8c6 d2d4 e7e6 e2e4 g8f6 d4d5 e6d5 e4d5 d8e7
bestmove g1f3

Doubled pawns and tempo
info depth 1 score cp 58 time 29 nodes 21 nps 708 pv b1c3
info depth 2 score cp 8 time 30 nodes 41 nps 2055 pv b1c3 b8c6
info depth 3 score cp 58 time 30 nodes 117 nps 5895 pv b1c3 b8c6 g1f3
info depth 4 score cp 8 time 32 nodes 140 nps 9786 pv b1c3 b8c6 g1f3 g8f6
info depth 5 score cp 51 time 34 nodes 1500 nps 52791 pv b1c3 b8c6 g1f3 g8f6 e2e4
info depth 6 score cp 8 time 46 nodes 8791 nps 230486 pv b1c3 b8c6 g1f3 g8f6 e2e4 e7e5
info depth 7 score cp 38 time 58 nodes 10695 nps 363213 pv b1c3 b8c6 g1f3 g8f6 e2e4 d7d5 d2d3
info depth 8 score cp 35 time 129 nodes 55218 nps 592857 pv b1c3 b8c6 d2d4 g8f6 c1f4 d7d5 c3b5 e7e5
info depth 9 score cp 33 time 305 nodes 108102 nps 603440 pv b1c3 b8c6 d2d4 g8f6 c1f4 e7e6 g1f3 d7d5 e2e3
info depth 10 score cp 24 time 1060 nodes 704160 nps 838001 pv e2e4 b8c6 d2d4 d7d5 e4e5 e7e6 b1c3 g8e7 g1f3 e7f5
info depth 11 score cp 34 time 1751 nodes 644262 nps 875389 pv e2e4 b8c6 d2d4 d7d5 e4e5 c8f5 b1c3 e7e6 g1f3 g8e7 f1b5
info depth 12 score cp 32 time 5273 nodes 3284355 nps 913525 pv g1f3 d7d5 d2d4 g8f6 e2e3 b8d7 f1d3 e7e6 e1g1 f8b4 c1d2
 d8e7
info depth 13 score cp 33 time 8868 nodes 3342069 nps 920065 pv g1f3 d7d5 d2d4 g8f6 e2e3 b8d7 b1c3 e7e6 f1b5 f8b4 e1g1
 e8g8 d1d3
info depth 14 score cp 28 time 21951 nodes 12262659 nps 930324 pv g1f3 d7d5 d2d4 g8f6 e2e3 b8d7 f1b5 e7e6 e1g1 f8d6
 b1c3 e8g8 d1d3 e6e5
info depth 14 score cp 28 time 60001 nodes 35898492 nps 938658 pv g1f3 d7d5 d2d4 g8f6 e2e3 b8d7 f1b5 e7e6 e1g1 f8d6
 b1c3 e8g8 d1d3 e6e5
bestmove g1f3


Backwards pawn + Isolated pawn + Passed Pawn + Tempo bug fix
info depth 1 score cp 42 time 22 nodes 21 nps 921 pv b1c3
info depth 2 score cp 8 time 23 nodes 60 nps 3494 pv b1c3 b8c6
info depth 3 score cp 42 time 23 nodes 138 nps 9373 pv b1c3 b8c6 g1f3
info depth 4 score cp 8 time 24 nodes 997 nps 48804 pv b1c3 b8c6 g1f3 g8f6
info depth 5 score cp 35 time 26 nodes 1322 nps 95525 pv b1c3 b8c6 g1f3 g8f6 e2e4
info depth 6 score cp 2 time 35 nodes 7171 nps 273446 pv b1c3 b8c6 g1f3 g8f6 e2e4 e7e5
info depth 7 score cp 16 time 45 nodes 9613 nps 426186 pv b1c3 b8c6 g1f3 g8f6 e2e4 d7d5 d2d3
info depth 8 score cp 4 time 121 nodes 68331 nps 719061 pv b1c3 g8f6 g1f3 d7d5 d2d4
info depth 9 score cp 22 time 356 nodes 208608 nps 831693 pv e2e4 b8c6 b1c3 g8f6 g1f3
info depth 10 score cp 20 time 598 nodes 219335 nps 862160 pv e2e4 b8c6 d2d4 d7d5 e4d5 d8d5 g1e2 e7e5 b1c3 f8b4
info depth 11 score cp 21 time 1160 nodes 499493 nps 874872 pv e2e4 b8c6 d2d4 e7e6 g1f3 d7d5 b1d2 d5e4 d2e4 f8b4 c2c3
 g8f6
info depth 12 score cp 25 time 3618 nodes 2230681 nps 896888 pv e2e4 e7e6 d2d4 d7d5 e4e5 b8c6 g1f3
info depth 13 score cp 24 time 8145 nodes 3821380 nps 867639 pv e2e4 e7e6 d2d4 d7d5 e4e5 b8c6 g1f3 g8e7 b1c3 c8d7 c1f4
 e7g6 f4g5
info depth 14 score cp 8 time 51731 nodes 39338946 nps 897064 pv e2e4 e7e5 g1f3 g8f6 b1c3 b8c6 f1b5
info depth 14 score cp 8 time 60000 nodes 7283936 nps 894820 pv e2e4 e7e5 g1f3 g8f6 b1c3 b8c6 f1b5
bestmove e2e4

QSearch depth made unlimited and switched update_search to use bitwise and instead of modulo
info depth 1 score cp 42 time 28 nodes 21 nps 731 pv b1c3
info depth 2 score cp 8 time 29 nodes 60 nps 2779 pv b1c3 b8c6
info depth 3 score cp 42 time 29 nodes 138 nps 7466 pv b1c3 b8c6 g1f3
info depth 4 score cp 8 time 31 nodes 997 nps 39190 pv b1c3 b8c6 g1f3 g8f6
info depth 5 score cp 35 time 32 nodes 1322 nps 77434 pv b1c3 b8c6 g1f3 g8f6 e2e4
info depth 6 score cp 2 time 42 nodes 7227 nps 230268 pv b1c3 b8c6 g1f3 g8f6 e2e4 e7e5
info depth 7 score cp 16 time 53 nodes 9952 nps 370391 pv b1c3 b8c6 g1f3 g8f6 d2d4 d7d5 e2e3
info depth 8 score cp 12 time 87 nodes 30351 nps 572821 pv b1c3 b8c6 g1f3 g8f6 e2e4 d7d5 e4e5 f6g4
info depth 9 score cp 13 time 254 nodes 142516 nps 757727 pv g1f3 g8f6 d2d4 d7d5 b1c3 b8c6
info depth 10 score cp 20 time 796 nodes 480822 nps 845007 pv e2e4 b8c6 d2d4 d7d5 e4d5 d8d5 g1e2 e7e5 b1c3 f8b4
info depth 11 score cp 20 time 2691 nodes 1654865 nps 865133 pv g1f3 b8c6 d2d4 e7e6 e2e3 g8e7 f1b5 d7d5 e1g1 a7a6 b5d3
info depth 12 score cp 14 time 5102 nodes 2121228 nps 872100 pv g1f3 g8f6 e2e3 b8c6 f1b5 e7e5 b1c3 e5e4 f3g5 d7d5 b5c6
 b7c6 e1g1
info depth 13 score cp 21 time 21564 nodes 14061295 nps 858387 pv e2e4 e7e5 b1c3 g8f6 g1f3 f8b4 f3e5 b4c3 d2c3 d7d6 e5f3
 f6e4 f3d4
info depth 13 score cp 21 time 60001 nodes 32408730 nps 848642 pv e2e4 e7e5 b1c3 g8f6 g1f3 f8b4 f3e5 b4c3 d2c3 d7d6 e5f3
 f6e4 f3d4
bestmove e2e4

king_positions variable instead of white and black king position
info depth 1 score cp 42 time 24 nodes 21 nps 853 pv b1c3
info depth 2 score cp 8 time 25 nodes 60 nps 3223 pv b1c3 b8c6
info depth 3 score cp 42 time 25 nodes 138 nps 8647 pv b1c3 b8c6 g1f3
info depth 4 score cp 8 time 26 nodes 997 nps 45162 pv b1c3 b8c6 g1f3 g8f6
info depth 5 score cp 35 time 29 nodes 1322 nps 86603 pv b1c3 b8c6 g1f3 g8f6 e2e4
info depth 6 score cp 2 time 38 nodes 7227 nps 253267 pv b1c3 b8c6 g1f3 g8f6 e2e4 e7e5
info depth 7 score cp 16 time 49 nodes 9952 nps 401576 pv b1c3 b8c6 g1f3 g8f6 d2d4 d7d5 e2e3
info depth 8 score cp 12 time 82 nodes 30351 nps 603329 pv b1c3 b8c6 g1f3 g8f6 e2e4 d7d5 e4e5 f6g4
info depth 9 score cp 13 time 243 nodes 142516 nps 791305 pv g1f3 g8f6 d2d4 d7d5 b1c3 b8c6
info depth 10 score cp 20 time 779 nodes 480822 nps 864263 pv e2e4 b8c6 d2d4 d7d5 e4d5 d8d5 g1e2 e7e5 b1c3 f8b4
info depth 11 score cp 20 time 2649 nodes 1654865 nps 878616 pv g1f3 b8c6 d2d4 e7e6 e2e3 g8e7 f1b5 d7d5 e1g1 a7a6 b5d3
info depth 12 score cp 14 time 5007 nodes 2121228 nps 888639 pv g1f3 g8f6 e2e3 b8c6 f1b5 e7e5 b1c3 e5e4 f3g5 d7d5 
b5c6 b7c6 e1g1
info depth 13 score cp 21 time 20717 nodes 14061295 nps 893494 pv e2e4 e7e5 b1c3 g8f6 g1f3 f8b4 f3e5 b4c3 d2c3 d7d6 
e5f3 f6e4 f3d4
info depth 13 score cp 21 time 60000 nodes 34493582 nps 883394 pv e2e4 e7e5 b1c3 g8f6 g1f3 f8b4 f3e5 b4c3 d2c3 d7d6 
e5f3 f6e4 f3d4
bestmove e2e4

History bug fixed
info depth 1 score cp 42 time 32 nodes 21 nps 641 pv b1c3
info depth 2 score cp 8 time 33 nodes 60 nps 2446 pv b1c3 b8c6
info depth 3 score cp 42 time 33 nodes 138 nps 6577 pv b1c3 b8c6 g1f3
info depth 4 score cp 8 time 34 nodes 997 nps 34826 pv b1c3 b8c6 g1f3 g8f6
info depth 5 score cp 35 time 36 nodes 1322 nps 69255 pv b1c3 b8c6 g1f3 g8f6 e2e4
info depth 6 score cp 2 time 45 nodes 7156 nps 211144 pv b1c3 b8c6 g1f3 g8f6 e2e4 e7e5
info depth 7 score cp 16 time 56 nodes 9979 nps 347334 pv b1c3 b8c6 g1f3 g8f6 d2d4 d7d5 e2e3
info depth 8 score cp 12 time 90 nodes 29440 nps 543658 pv b1c3 b8c6 g1f3 g8f6 e2e4 d7d5 e4e5 f6g4
info depth 9 score cp 20 time 175 nodes 75861 nps 711260 pv b1c3 b8c6 g1f3 d7d5 e2e4 d5e4 c3e4 g8f6 f3g5
info depth 10 score cp 8 time 613 nodes 381581 nps 825980 pv b1c3 b8c6 g1f3 d7d5 d2d4 g8f6 e2e3 e7e6 f1b5 f8b4
info depth 11 score cp 9 time 1576 nodes 852019 nps 861605 pv b1c3 d7d5 g1f3 g8f6 e2e3 e7e6 f1d3 b8c6 e1g1 f8b4 d3b5
info depth 12 score cp 18 time 6737 nodes 4496815 nps 869034 pv e2e4 e7e5 g1f3 g8f6 b1c3 f8b4 f3e5 d7d6 e5d3 b4c3 d2c3
 f6e4
info depth 13 score cp 11 time 17172 nodes 8951968 nps 862287 pv e2e4 e7e5 g1f3 g8f6 b1c3 b8c6 f1b5 f8b4 d1e2 d8e7 b5c6
 d7c6 e1g1
info depth 13 score cp 11 time 60000 nodes 36785290 nps 859876 pv e2e4 e7e5 g1f3 g8f6 b1c3 b8c6 f1b5 f8b4 d1e2 d8e7 b5c6
 d7c6 e1g1
bestmove e2e4

Bishop pair bonus + optimizations
info depth 1 score cp 42 time 10 nodes 21 nps 2083 pv b1c3
info depth 2 score cp 8 time 10 nodes 60 nps 7712 pv b1c3 b8c6
info depth 3 score cp 42 time 10 nodes 138 nps 20488 pv b1c3 b8c6 g1f3
info depth 4 score cp 8 time 12 nodes 1000 nps 98275 pv b1c3 b8c6 g1f3 g8f6
info depth 5 score cp 35 time 14 nodes 1325 nps 178525 pv b1c3 b8c6 g1f3 g8f6 e2e4
info depth 6 score cp 2 time 24 nodes 7153 nps 402002 pv b1c3 b8c6 g1f3 g8f6 e2e4 e7e5
info depth 7 score cp 16 time 35 nodes 10004 nps 551832 pv b1c3 b8c6 g1f3 g8f6 d2d4 d7d5 e2e3
info depth 8 score cp 12 time 69 nodes 29465 nps 703781 pv b1c3 b8c6 g1f3 g8f6 e2e4 d7d5 e4e5 f6g4
info depth 9 score cp 20 time 161 nodes 81745 nps 808891 pv b1c3 b8c6 g1f3 d7d5 e2e4 d5e4 c3e4 g8f6 f3g5
info depth 10 score cp 22 time 751 nodes 533342 nps 883664 pv e2e4 b8c6 g1f3 g8f6 b1c3
info depth 11 score cp 29 time 1959 nodes 1100658 nps 900560 pv e2e4 e7e6 b1c3 d7d5 d2d4 g8e7 e4e5 b8c6 g1f3 c8d7 f1b5
info depth 12 score cp 26 time 3889 nodes 1773146 nps 909566 pv e2e4 d7d5 e4d5 g8f6 d2d4 f6d5 g1f3 e7e6 c2c4 d5f6 b1c3
 b8c6
info depth 13 score cp 24 time 10947 nodes 6387008 nps 906635 pv e2e4 e7e6 g1f3 d7d5 e4d5 e6d5 f1b5 b8c6 e1g1 a7a6 b5c6
 b7c6 d2d4 g8f6 b1c3
info depth 13 score cp 24 time 60000 nodes 44049689 nps 899564 pv e2e4 e7e6 g1f3 d7d5 e4d5 e6d5 f1b5 b8c6 e1g1 a7a6 b5c6
 b7c6 d2d4 g8f6 b1c3
bestmove e2e4

piece lists
info depth 1 score cp 42 time 4 nodes 21 nps 4288 pv b1c3
info depth 2 score cp 8 time 5 nodes 60 nps 15390 pv b1c3 b8c6
info depth 3 score cp 42 time 5 nodes 138 nps 40303 pv b1c3 b8c6 g1f3
info depth 4 score cp 8 time 6 nodes 1000 nps 174244 pv b1c3 b8c6 g1f3 g8f6
info depth 5 score cp 35 time 8 nodes 1325 nps 293907 pv b1c3 b8c6 g1f3 g8f6 e2e4
info depth 6 score cp 2 time 18 nodes 7159 nps 537947 pv b1c3 b8c6 g1f3 g8f6 e2e4 e7e5
info depth 7 score cp 16 time 28 nodes 10003 nps 687554 pv b1c3 b8c6 g1f3 g8f6 d2d4 d7d5 e2e3
info depth 8 score cp 12 time 63 nodes 29520 nps 778892 pv b1c3 b8c6 g1f3 g8f6 e2e4 d7d5 e4e5 f6g4
info depth 9 score cp 20 time 157 nodes 81784 nps 829304 pv b1c3 b8c6 g1f3 d7d5 e2e4 d5e4 c3e4 g8f6 f3g5
info depth 10 score cp 22 time 769 nodes 533813 nps 864509 pv e2e4 b8c6 g1f3 g8f6 b1c3
info depth 11 score cp 29 time 2020 nodes 1101444 nps 874086 pv e2e4 e7e6 b1c3 d7d5 d2d4 g8e7 e4e5 b8c6 g1f3 c8d7 f1b5
info depth 12 score cp 26 time 3989 nodes 1756573 nps 882986 pv e2e4 d7d5 e4d5 g8f6 d2d4 f6d5 g1f3 e7e6 c2c4 d5f6 b1c3
 b8c6
info depth 13 score cp 24 time 11534 nodes 6433972 nps 863240 pv e2e4 e7e6 g1f3 d7d5 e4d5 e6d5 f1b5 b8c6 e1g1 a7a6 b5c6
 b7c6 d2d4 g8f6 b1c3
info depth 13 score cp 24 time 60002 nodes 39771322 nps 828769 pv e2e4 e7e6 g1f3 d7d5 e4d5 e6d5 f1b5 b8c6 e1g1 a7a6 b5c6
 b7c6 d2d4 g8f6 b1c3
bestmove e2e4


PST optimizations
info depth 1 score cp 42 time 20 nodes 21 nps 1039 pv b1c3
info depth 2 score cp 8 time 20 nodes 124 nps 5999 pv g1f3 b8c6
info depth 3 score cp 42 time 20 nodes 260 nps 12471 pv g1f3 b8c6 b1c3
info depth 4 score cp 6 time 22 nodes 1278 nps 56853 pv g1f3 b8c6 b1c3 e7e5
info depth 5 score cp 36 time 24 nodes 2761 nps 113248 pv g1f3 b8c6 b1c3 e7e5 e2e4
info depth 6 score cp 2 time 40 nodes 16142 nps 398036 pv g1f3 g8f6 b1c3 b8c6
info depth 7 score cp 16 time 50 nodes 25591 nps 501865 pv g1f3 g8f6 b1c3 b8c6 d2d4 d7d5 e2e3
info depth 8 score cp 14 time 105 nodes 71794 nps 677424 pv b1c3 b8c6 g1f3 g8f6
info depth 9 score cp 22 time 381 nodes 280956 nps 735938 pv e2e4 b8c6 b1c3 g8f6
info depth 10 score cp 12 time 930 nodes 743629 nps 799256 pv e2e4 d7d5 e4d5 g8f6 g1f3 f6d5 d2d4 b8c6 c2c4 d5f6
info depth 11 score cp 26 time 1372 nodes 1126310 nps 820458 pv e2e4 d7d5 e4d5 g8f6 g1f3 f6d5 d2d4 d8d6 c2c4 d5f4 b1c3
info depth 12 score cp 32 time 2525 nodes 2100240 nps 831646 pv e2e4 d7d5 e4d5 g8f6 g1f3 f6d5 c2c4 d5f4 d2d4 d8d6
info depth 13 score cp 29 time 9886 nodes 8337340 nps 843269 pv e2e4 b8c6 d2d4 e7e6 g1f3 d7d5 e4e5 g8e7 b1c3 c8d7 c1f4
 e7g6 f4g5
info depth 13 score cp 29 time 60000 nodes 48060624 nps 801001 pv e2e4 b8c6 d2d4 e7e6 g1f3 d7d5 e4e5 g8e7 b1c3 c8d7 c1f4
 e7g6 f4g5
bestmove e2e4

Rook semi open file and open file bonuses
info depth 1 score cp 42 time 7 nodes 21 nps 2964 pv b1c3
info depth 2 score cp 8 time 7 nodes 124 nps 16524 pv g1f3 b8c6
info depth 3 score cp 42 time 7 nodes 260 nps 33774 pv g1f3 b8c6 b1c3
info depth 4 score cp 6 time 9 nodes 1276 nps 135801 pv g1f3 b8c6 b1c3 e7e5
info depth 5 score cp 36 time 11 nodes 2758 nps 242803 pv g1f3 b8c6 b1c3 e7e5 e2e4
info depth 6 score cp 2 time 28 nodes 16142 nps 575126 pv g1f3 g8f6 b1c3 b8c6
info depth 7 score cp 16 time 38 nodes 25592 nps 657672 pv g1f3 g8f6 b1c3 b8c6 d2d4 d7d5 e2e3
info depth 8 score cp 14 time 94 nodes 71761 nps 755522 pv b1c3 b8c6 g1f3 g8f6
info depth 9 score cp 22 time 345 nodes 280437 nps 811532 pv e2e4 b8c6 b1c3 g8f6
info depth 10 score cp 12 time 846 nodes 693219 nps 818762 pv e2e4 d7d5 e4d5 g8f6 d2d4 f6d5 c2c4 d5f6 b1c3 b8c6
info depth 11 score cp 34 time 1898 nodes 1561747 nps 822666 pv e2e4 e7e6 b1c3 d7d5 d2d4 g8e7 g1f3 b8c6 e4e5 c8d7 c1f4
info depth 12 score cp 32 time 3373 nodes 2788781 nps 826760 pv e2e4 e7e6 b1c3 d7d5 d2d4 g8e7 g1f3 b8c6 e4e5 c8d7 c1f4
 a8c8
info depth 13 score cp 29 time 8402 nodes 6966813 nps 829100 pv e2e4 e7e6 d2d4 d7d5 e4e5 b8c6 b1c3
info depth 13 score cp 29 time 60001 nodes 47787229 nps 796440 pv e2e4 e7e6 d2d4 d7d5 e4e5 b8c6 b1c3
bestmove e2e4

adaptive null move pruning reduction change
info depth 1 score cp 42 time 9 nodes 21 nps 2192 pv b1c3
info depth 2 score cp 8 time 9 nodes 124 nps 12400 pv g1f3 b8c6
info depth 3 score cp 42 time 10 nodes 260 nps 25548 pv g1f3 b8c6 b1c3
info depth 4 score cp 6 time 11 nodes 1276 nps 107470 pv g1f3 b8c6 b1c3 e7e5
info depth 5 score cp 36 time 13 nodes 2758 nps 200744 pv g1f3 b8c6 b1c3 e7e5 e2e4
info depth 6 score cp 2 time 29 nodes 16142 nps 547394 pv g1f3 g8f6 b1c3 b8c6
info depth 7 score cp 16 time 39 nodes 25553 nps 644449 pv g1f3 g8f6 b1c3 b8c6 d2d4 d7d5 e2e3
info depth 8 score cp 14 time 94 nodes 73369 nps 777823 pv b1c3 b8c6 g1f3 g8f6
info depth 9 score cp 22 time 361 nodes 305569 nps 846413 pv e2e4 b8c6 b1c3 g8f6
info depth 10 score cp 22 time 657 nodes 561788 nps 854230 pv e2e4 b8c6 b1c3 g8f6 g1f3 d7d5 e4e5 f6e4 c3e4 d5e4
info depth 11 score cp 33 time 1809 nodes 1544547 nps 853733 pv e2e4 b8c6 g1f3 g8f6 e4e5 f6d5 b1c3 d5c3 d2c3 d7d5 c1f4
info depth 12 score cp 32 time 2987 nodes 2593277 nps 868007 pv e2e4 b8c6 g1f3 e7e6 b1c3 d7d5 d2d4 g8e7 e4e5 c8d7 c1f4
 a8c8
info depth 13 score cp 29 time 7342 nodes 6417711 nps 874044 pv e2e4 b8c6 g1f3 e7e6 d2d4 d7d5 e4e5 g8e7 b1c3
info depth 13 score cp 29 time 60000 nodes 49319159 nps 821973 pv e2e4 b8c6 g1f3 e7e6 d2d4 d7d5 e4e5 g8e7 b1c3
bestmove e2e4

King eval + bug fix
info depth 1 score cp 42 time 0 nodes 21 nps 210000 pv b1c3
info depth 2 score cp 8 time 0 nodes 124 nps 436686 pv g1f3 b8c6
info depth 3 score cp 42 time 0 nodes 260 nps 516833 pv g1f3 b8c6 b1c3
info depth 4 score cp 6 time 2 nodes 1276 nps 443812 pv g1f3 b8c6 b1c3 e7e5
info depth 5 score cp 36 time 5 nodes 2758 nps 534486 pv g1f3 b8c6 b1c3 e7e5 e2e4
info depth 6 score cp 2 time 24 nodes 16144 nps 660903 pv g1f3 g8f6 b1c3 b8c6
info depth 7 score cp 16 time 36 nodes 25615 nps 696969 pv g1f3 g8f6 b1c3 b8c6 d2d4 d7d5 e2e3
info depth 8 score cp 14 time 107 nodes 81305 nps 759838 pv b1c3 g8f6 e2e4 b8c6 g1f3 d7d5
info depth 9 score cp 20 time 425 nodes 343014 nps 805540 pv e2e4 b8c6 g1f3 d7d5 b1c3 d5e4
info depth 10 score cp 12 time 1051 nodes 851211 nps 809691 pv e2e4 d7d5 e4d5 g8f6 g1f3 f6d5 d2d4 b8c6 c2c4 d5f6
info depth 11 score cp 26 time 1688 nodes 1372168 nps 812559 pv e2e4 d7d5 e4d5 g8f6 g1f3 f6d5 d2d4 d8d6 c2c4 d5f4 b1c3
info depth 12 score cp 32 time 3644 nodes 2980064 nps 817622 pv e2e4 d7d5 e4d5 g8f6 g1f3 f6d5 d2d4 d8d6 c2c4 d5f4 f3e5
 d6f6
info depth 13 score cp 34 time 30808 nodes 24618401 nps 799081 pv e2e4 e7e6 d2d4 d7d5 e4d5 e6d5 g1f3 b8c6
info depth 13 score cp 34 time 60000 nodes 47359192 nps 789307 pv e2e4 e7e6 d2d4 d7d5 e4d5 e6d5 g1f3 b8c6
bestmove e2e4

Qsearch bug fix
info depth 1 score cp 42 time 0 nodes 21 nps 210000 pv b1c3
info depth 2 score cp 8 time 0 nodes 124 nps 230537 pv g1f3 b8c6
info depth 3 score cp 42 time 0 nodes 260 nps 346306 pv g1f3 b8c6 b1c3
info depth 4 score cp 6 time 2 nodes 1276 nps 461015 pv g1f3 b8c6 b1c3 e7e5
info depth 5 score cp 36 time 4 nodes 2758 nps 554469 pv g1f3 b8c6 b1c3 e7e5 e2e4
info depth 6 score cp 2 time 20 nodes 14481 nps 722285 pv g1f3 g8f6 b1c3 b8c6
info depth 7 score cp 16 time 31 nodes 23706 nps 760634 pv g1f3 g8f6 b1c3 b8c6 d2d4 d7d5 e2e3
info depth 8 score cp 14 time 88 nodes 70464 nps 792913 pv b1c3 b8c6 g1f3 g8f6
info depth 9 score cp 20 time 359 nodes 299418 nps 832241 pv e2e4 b8c6 g1f3 d7d5 b1c3 d5e4
info depth 10 score cp 30 time 776 nodes 643230 nps 828746 pv e2e4 b8c6 d2d4 e7e6 b1c3 d7d5 g1f3 g8f6 e4e5
info depth 11 score cp 20 time 2003 nodes 1663366 nps 830143 pv e2e4 d7d5 e4d5 g8f6 f1b5 c8d7 b5d7 b8d7 b1c3 d7b6 g1f3
 a8c8 e1g1
info depth 12 score cp 32 time 3412 nodes 2854058 nps 836374 pv e2e4 d7d5 e4d5 g8f6 g1f3 f6d5 d2d4 b8c6 c2c4 d5f6 b1c3
 e7e6
info depth 13 score cp 33 time 27309 nodes 21979170 nps 804818 pv e2e4 b8c6 g1f3 e7e5 d2d4 e5d4 f3d4 g8f6 b1c3 f8c5 c1e3
 c6d4 e3d4
bestmove e2e4

Reverse Futility Pruning
info depth 1 score cp 42 time 0 nodes 21 nps 210000 pv b1c3
info depth 2 score cp 8 time 0 nodes 124 nps 203638 pv g1f3 b8c6
info depth 3 score cp 42 time 0 nodes 260 nps 301331 pv g1f3 b8c6 b1c3
info depth 4 score cp 6 time 3 nodes 1276 nps 399994 pv g1f3 b8c6 b1c3 e7e5
info depth 5 score cp 36 time 5 nodes 2622 nps 489102 pv g1f3 b8c6 b1c3 e7e5 e2e4
info depth 6 score cp 2 time 22 nodes 14154 nps 635251 pv g1f3 g8f6 b1c3 b8c6
info depth 7 score cp 16 time 34 nodes 23063 nps 668301 pv g1f3 g8f6 b1c3 b8c6 d2d4 d7d5 e2e3
info depth 8 score cp 14 time 94 nodes 67314 nps 715976 pv b1c3 b8c6 g1f3 g8f6
info depth 9 score cp 20 time 361 nodes 279857 nps 774390 pv e2e4 b8c6 g1f3 d7d5 b1c3 d5e4
info depth 10 score cp 17 time 814 nodes 640481 nps 786301 pv e2e4 e7e5 b1c3 b8c6 g1f3
info depth 11 score cp 22 time 1838 nodes 1459188 nps 793553 pv e2e4 d7d5 e4d5 g8f6 b1c3 f6d5 c3d5 d8d5 d2d4 b8c6 g1f3
info depth 12 score cp 32 time 2880 nodes 2303319 nps 799618 pv e2e4 d7d5 e4d5 g8f6 g1f3 f6d5 d2d4 b8c6 c2c4 d5f6 b1c3
 e7e6
info depth 13 score cp 28 time 16226 nodes 12681408 nps 781541 pv e2e4 e7e6 g1f3 d7d5 e4d5 e6d5 f1b5 b8d7 e1g1 g8f6 d2d4
 f8e7 b1c3 e8g8
info depth 13 score cp 28 time 60000 nodes 45028519 nps 750466 pv e2e4 e7e6 g1f3 d7d5 e4d5 e6d5 f1b5 b8d7 e1g1 g8f6 d2d4
 f8e7 b1c3 e8g8
bestmove e2e4

Removal of PSTs in move ordering
info depth 1 score cp 42 time 0 nodes 26 nps 260000 pv b1c3
info depth 2 score cp 8 time 0 nodes 129 nps 203407 pv g1f3 b8c6
info depth 3 score cp 42 time 0 nodes 262 nps 294692 pv g1f3 b8c6 b1c3
info depth 4 score cp 6 time 3 nodes 1325 nps 395886 pv g1f3 b8c6 b1c3 e7e5
info depth 5 score cp 36 time 6 nodes 2654 nps 437531 pv g1f3 b8c6 b1c3 e7e5 e2e4
info depth 6 score cp 2 time 29 nodes 16150 nps 544355 pv g1f3 g8f6 b1c3 b8c6
info depth 7 score cp 16 time 42 nodes 23980 nps 569364 pv g1f3 g8f6 b1c3 b8c6 d2d4 d7d5 e2e3
info depth 8 score cp 14 time 134 nodes 80027 nps 596771 pv b1c3 b8c6 g1f3 g8f6
info depth 9 score cp 20 time 258 nodes 158388 nps 611702 pv b1c3 b8c6 g1f3 d7d5 e2e4 d5e4 c3e4 g8f6 f3g5
info depth 10 score cp 12 time 1094 nodes 701028 nps 640493 pv e2e4 d7d5 e4d5 g8f6 g1f3 f6d5 d2d4 b8c6 c2c4 d5f6
info depth 11 score cp 22 time 1634 nodes 1062151 nps 650016 pv e2e4 d7d5 e4d5 g8f6 b1c3 f6d5 c3d5 d8d5 d2d4 b8c6 g1f3
info depth 12 score cp 32 time 3060 nodes 2030084 nps 663311 pv e2e4 d7d5 e4d5 g8f6 g1f3 f6d5 d2d4 e7e6 c2c4 d5f6 b1c3
 b8c6
info depth 13 score cp 29 time 13259 nodes 8880216 nps 669717 pv e2e4 e7e6 g1f3 b8c6
info depth 13 score cp 29 time 60000 nodes 38603019 nps 643374 pv e2e4 e7e6 g1f3 b8c6
bestmove e2e4

Reverse Futility margin increased from 100 -> 150
info depth 1 score cp 42 time 0 nodes 26 nps 260000 pv b1c3
info depth 2 score cp 8 time 0 nodes 129 nps 235450 pv g1f3 b8c6
info depth 3 score cp 42 time 0 nodes 262 nps 346112 pv g1f3 b8c6 b1c3
info depth 4 score cp 6 time 2 nodes 1325 nps 465409 pv g1f3 b8c6 b1c3 e7e5
info depth 5 score cp 36 time 5 nodes 2793 nps 537839 pv g1f3 b8c6 b1c3 e7e5 e2e4
info depth 6 score cp 2 time 22 nodes 14820 nps 668109 pv g1f3 g8f6 b1c3 b8c6
info depth 7 score cp 16 time 33 nodes 23994 nps 714896 pv g1f3 g8f6 b1c3 b8c6 d2d4 d7d5 e2e3
info depth 8 score cp 14 time 96 nodes 72296 nps 750021 pv b1c3 b8c6 g1f3 g8f6
info depth 9 score cp 20 time 224 nodes 170376 nps 759068 pv b1c3 b8c6 g1f3 d7d5 e2e4 d5e4 c3e4 g8f6 f3g5
info depth 10 score cp 12 time 938 nodes 726482 nps 774175 pv e2e4 d7d5 e4d5 g8f6 g1f3 f6d5 d2d4 b8c6 c2c4 d5f6
info depth 11 score cp 22 time 1470 nodes 1159295 nps 788391 pv e2e4 d7d5 e4d5 g8f6 b1c3 f6d5 c3d5 d8d5 d2d4 b8c6 g1f3
info depth 12 score cp 32 time 2733 nodes 2181641 nps 798232 pv e2e4 d7d5 e4d5 g8f6 g1f3 f6d5 d2d4 e7e6 c2c4 d5f6 b1c3 
b8c6
info depth 13 score cp 33 time 13974 nodes 11229348 nps 803565 pv e2e4 g8f6 b1c3 e7e5 g1f3 b8c6 d2d4 e5d4 f3d4 f8b4 d4c6
 b4c3 b2c3 d7c6
info depth 13 score cp 33 time 60000 nodes 47093958 nps 784887 pv e2e4 g8f6 b1c3 e7e5 g1f3 b8c6 d2d4 e5d4 f3d4 f8b4 d4c6
 b4c3 b2c3 d7c6
bestmove e2e4

New implementation of LMR with PVS
info depth 1 score cp 42 time 0 nodes 26 nps 260000 pv b1c3
info depth 2 score cp 8 time 0 nodes 129 nps 243942 pv g1f3 b8c6
info depth 3 score cp 42 time 0 nodes 262 nps 357950 pv g1f3 b8c6 b1c3
info depth 4 score cp 36 time 1 nodes 1032 nps 519630 pv d2d4 b8c6 b1c3
info depth 5 score cp 8 time 7 nodes 4548 nps 636259 pv b1c3 d7d5 g1f3 g8f6
info depth 6 score cp 20 time 18 nodes 12809 nps 701139 pv b1c3 b8c6 e2e3 e7e5 g1f3
info depth 7 score cp 16 time 39 nodes 29380 nps 737635 pv b1c3 b8c6 g1f3 g8f6 d2d4 d7d5 e2e3
info depth 8 score cp 16 time 99 nodes 79235 nps 797004 pv b1c3 g8f6 d2d4 d7d5 e2e3 b8c6 g1f3
info depth 9 score cp 14 time 207 nodes 168941 nps 812786 pv b1c3 g8f6 e2e4 b8c6 g1f3 d7d5 e4e5 f6g4
info depth 10 score cp 25 time 589 nodes 492122 nps 834874 pv e2e4 b8c6 b1c3 e7e5 g1e2 g8f6 d2d4
info depth 11 score cp 22 time 1713 nodes 1457115 nps 850391 pv e2e4 e7e5 b1c3 g8f6 g1f3 b8c6
info depth 12 score cp 29 time 3794 nodes 3210531 nps 846169 pv e2e4 b8c6
info depth 13 score cp 28 time 8952 nodes 7574085 nps 845997 pv e2e4 g8f6 e4e5 f6d5 d2d4 b8c6 c2c4 d5b6 g1f3 d7d5 b1d2
 e7e6 f1e2
info depth 13 score cp 28 time 60000 nodes 49078424 nps 817964 pv e2e4 g8f6 e4e5 f6d5 d2d4 b8c6 c2c4 d5b6 g1f3 d7d5 b1d2
 e7e6 f1e2
bestmove e2e4

crazy LMR
info depth 1 score cp 42 time 0 nodes 26 nps 260000 pv b1c3
info depth 2 score cp 8 time 0 nodes 129 nps 252362 pv g1f3 b8c6
info depth 3 score cp 42 time 0 nodes 460 nps 519768 pv g1f3 b8c6 b1c3
info depth 4 score cp 6 time 2 nodes 1591 nps 578562 pv g1f3 b8c6 b1c3 e7e5
info depth 5 score cp 8 time 6 nodes 4050 nps 624611 pv g1f3 d7d5 b1c3 g8f6
info depth 6 score cp 22 time 15 nodes 10477 nps 675537 pv g1f3 d7d5 e2e3 g8f6 b1c3
info depth 7 score cp 16 time 34 nodes 25531 nps 744994 pv g1f3 d7d5 b1c3 g8f6 d2d4 b8c6 e2e3
info depth 8 score cp 8 time 75 nodes 58735 nps 780854 pv e2e3 b8c6 g1f3 d7d6 b1c3 g8f6
info depth 9 score cp 13 time 198 nodes 161906 nps 813854 pv b1c3 g8f6 g1f3 d7d5 d2d4 b8c6 e2e3
info depth 10 score cp 27 time 347 nodes 282638 nps 814354 pv e2e4 d7d5 e4d5 d8d5 b1c3 d5e6 g1e2 g8f6 d2d4 b8c6
info depth 11 score cp 22 time 608 nodes 502852 nps 825827 pv e2e4 d7d5 e4d5 g8f6 b1c3 f6d5 c3d5 d8d5 d2d4 b8c6 g1f3
info depth 12 score cp 24 time 1436 nodes 1204456 nps 838635 pv e2e4 b8c6 d2d4 d7d5 e4d5 d8d5 g1f3 c8f5 b1c3 d5e6 c1e3
 g8f6
info depth 13 score cp 32 time 2618 nodes 2185385 nps 834575 pv e2e4 b8c6 b1c3 e7e6 d2d4 d7d5 e4e5 f8b4 g1e2 g8e7 a2a3
 b4a5
info depth 14 score cp 30 time 4905 nodes 4063341 nps 828314 pv e2e4 b8c6 d2d4 e7e6 g1f3 g8f6 b1d2 d7d5 e4e5 f6d7 f1b5
 f8b4
info depth 15 score cp 29 time 12103 nodes 9958208 nps 822736 pv e2e4 e7e6 d2d4 d7d5 b1c3 b8c6 g1e2 g8e7 e4e5 c8d7 c1g5
info depth 15 score cp 29 time 60000 nodes 47958193 nps 799296 pv e2e4 e7e6 d2d4 d7d5 b1c3 b8c6 g1e2 g8e7 e4e5 c8d7 c1g5
bestmove e2e4

FREE ROBUX
info depth 1 score cp 42 time 0 nodes 26 nps 260000 pv b1c3
info depth 2 score cp 8 time 0 nodes 129 nps 238775 pv g1f3 b8c6
info depth 3 score cp 42 time 0 nodes 365 nps 414658 pv g1f3 b8c6 b1c3
info depth 4 score cp 6 time 3 nodes 1569 nps 484671 pv g1f3 b8c6 b1c3 e7e5
info depth 5 score cp 36 time 6 nodes 3313 nps 549781 pv g1f3 b8c6 b1c3 e7e5 e2e4
info depth 6 score cp 2 time 29 nodes 18877 nps 640825 pv g1f3 b8c6 b1c3 g8f6 e2e4 e7e5
info depth 7 score cp 16 time 39 nodes 25947 nps 653603 pv g1f3 b8c6 b1c3 g8f6 d2d4 d7d5 e2e3
info depth 8 score cp 14 time 87 nodes 62845 nps 715024 pv b1c3 b8c6 g1f3
info depth 9 score cp 28 time 270 nodes 202779 nps 748658 pv e2e4 b8c6 b1c3 g8f6
info depth 10 score cp 12 time 509 nodes 387624 nps 760993 pv e2e4 d7d5 e4d5 g8f6 g1f3 f6d5 d2d4 b8c6 c2c4 d5f6
info depth 11 score cp 34 time 1324 nodes 998604 nps 753801 pv e2e4 b8c6 d2d4 e7e6 b1c3 d7d5 g1f3 g8e7 e4e5 c8d7 c1g5
info depth 12 score cp 32 time 2001 nodes 1522649 nps 760601 pv e2e4 b8c6 d2d4 e7e6 b1c3 d7d5 g1f3 g8e7 e4e5 c8d7 c1g5
 a8c8
info depth 13 score cp 33 time 2900 nodes 2214898 nps 763695 pv e2e4 b8c6 d2d4 e7e6 g1f3 d7d5 e4e5 c8d7 b1c3 f8b4 f1d3
 b4c3 b2c3 g8e7
info depth 14 score cp 35 time 33619 nodes 24820437 nps 738279 pv e2e4 b8c6 d2d4 e7e6 g1f3 g8f6 d4d5 e6d5 e4d5 c6b4 b1c3
 d7d6 f1b5 c8d7 d1e2
bestmove e2e4

checking material draws and reforming eval for future. rip nps ;C
info depth 1 score cp 42 time 0 nodes 26 nps 260000 pv b1c3
info depth 2 score cp 8 time 0 nodes 129 nps 237101 pv g1f3 b8c6
info depth 3 score cp 42 time 0 nodes 365 nps 408245 pv g1f3 b8c6 b1c3
info depth 4 score cp 6 time 3 nodes 1569 nps 484992 pv g1f3 b8c6 b1c3 e7e5
info depth 5 score cp 36 time 6 nodes 3309 nps 548575 pv g1f3 b8c6 b1c3 e7e5 e2e4
info depth 6 score cp 2 time 32 nodes 20373 nps 631404 pv g1f3 b8c6 b1c3 g8f6 e2e4 e7e5
info depth 7 score cp 16 time 43 nodes 27535 nps 638120 pv g1f3 b8c6 b1c3 g8f6 d2d4 d7d5 e2e3
info depth 8 score cp 14 time 98 nodes 65512 nps 666307 pv b1c3 b8c6 g1f3
info depth 9 score cp 23 time 170 nodes 114556 nps 671024 pv b1c3 b8c6 e2e4 g8f6 g1f3 d7d5
info depth 10 score cp 23 time 252 nodes 170313 nps 673301 pv b1c3 b8c6 e2e4 g8f6 g1f3 d7d5 e4e5 f6e4 c3e4 d5e4
info depth 11 score cp 18 time 1459 nodes 1012376 nps 693433 pv e2e4 d7d5 e4d5 g8f6 f1b5 c8d7 b5c4 b7b5 c4b3 d7g4 f2f3
 g4f5
info depth 12 score cp 21 time 1861 nodes 1298897 nps 697858 pv e2e4 d7d5 e4d5 g8f6 f1b5 c8d7 b5c4 d7g4 f2f3 g4f5 b1c3
 c7c6 d5c6
info depth 13 score cp 29 time 5742 nodes 4011291 nps 698534 pv e2e4 e7e6 b1c3 d7d5 d2d4 g8e7 g1f3 b8c6 e4e5 c8d7 c1g5
 h7h6 g5f4
info depth 13 score cp 29 time 60001 nodes 41132264 nps 685517 pv e2e4 e7e6 b1c3 d7d5 d2d4 g8e7 g1f3 b8c6 e4e5 c8d7 c1g5
 h7h6 g5f4
bestmove e2e4

King tropism stuff and king safety stuff
info depth 1 score cp 54 time 0 nodes 26 nps 260000 pv b1c3
info depth 2 score cp 8 time 0 nodes 88 nps 175928 pv b1c3 g8f6
info depth 3 score cp 53 time 0 nodes 378 nps 408094 pv b1c3 b8c6 g1f3
info depth 4 score cp 8 time 3 nodes 1595 nps 456337 pv b1c3 b8c6 g1f3 g8f6
info depth 5 score cp 41 time 6 nodes 3343 nps 528696 pv b1c3 b8c6 g1f3 g8f6 e2e4
info depth 6 score cp 1 time 22 nodes 13657 nps 596366 pv b1c3 b8c6 g1f3 g8f6 e2e4 e7e5
info depth 7 score cp 18 time 32 nodes 20238 nps 614725 pv b1c3 b8c6 g1f3 g8f6 d2d4 d7d5 c1g5
info depth 8 score cp 11 time 79 nodes 50408 nps 633996 pv b1c3 b8c6 g1f3 g8f6 e2e4 d7d5 e4e5 f6g4
info depth 9 score cp 21 time 181 nodes 118775 nps 654028 pv b1c3 d7d5 e2e4 d5e4 c3e4 b8c6 g1f3
info depth 10 score cp 22 time 1089 nodes 732285 nps 671837 pv g1f3 g8f6 b1c3 d7d5 e2e3 b8c6
info depth 11 score cp 16 time 1683 nodes 1124737 nps 667939 pv e2e4 d7d5 e4e5 d5d4 g1f3 b8c6 d2d3 c8g4 d1e2 e7e6 b1d2
info depth 12 score cp 27 time 2806 nodes 1869697 nps 666121 pv e2e4 d7d5 e4d5 g8f6 d2d4 f6d5 c2c4 d5f6 b1c3 b8c6 g1f3
 c8g4
info depth 13 score cp 35 time 4951 nodes 3290422 nps 664483 pv e2e4 b8c6 b1c3 g8f6 d2d4 d7d5 e4e5 f6e4 c3e4 d5e4 g1e2
 c8g4 c1e3
info depth 14 score cp 36 time 7844 nodes 5206446 nps 663735 pv e2e4 b8c6 d2d4 d7d5 e4d5 d8d5 g1f3 c8g4 b1c3 d5e6 c1e3
 e8c8 d4d5 e6f6 f1b5
info depth 15 score cp 22 time 21191 nodes 14075016 nps 664191 pv e2e4 e7e5 b1c3 g8f6 g1f3 f8b4 a2a3 b4c3 d2c3 e8g8 f1c4
 f6e4 c4d5 e4f6 f3e5
info depth 15 score cp 22 time 60001 nodes 39633108 nps 660536 pv e2e4 e7e5 b1c3 g8f6 g1f3 f8b4 a2a3 b4c3 d2c3 e8g8 f1c4
 f6e4 c4d5 e4f6 f3e5
bestmove e2e4

info depth 1 score cp 54 time 0 nodes 26 nps 260000 pv b1c3
info depth 2 score cp 8 time 0 nodes 88 nps 169233 pv b1c3 g8f6
info depth 3 score cp 53 time 0 nodes 378 nps 387261 pv b1c3 b8c6 g1f3
info depth 4 score cp 8 time 3 nodes 1550 nps 454659 pv b1c3 b8c6 g1f3 g8f6
info depth 5 score cp 41 time 6 nodes 3221 nps 516925 pv b1c3 b8c6 g1f3 g8f6 e2e4
info depth 6 score cp 1 time 22 nodes 12875 nps 573010 pv b1c3 b8c6 g1f3 g8f6 e2e4 e7e5
info depth 7 score cp 18 time 32 nodes 19157 nps 581061 pv b1c3 b8c6 g1f3 g8f6 d2d4 d7d5 c1g5
info depth 8 score cp 11 time 110 nodes 66753 nps 603803 pv e2e4 b8c6 b1c3 g8f6
info depth 9 score cp 26 time 153 nodes 93834 nps 610771 pv e2e4 b8c6 b1c3 g8f6 g1f3 d7d5 e4e5 f6e4 c3e4
info depth 10 score cp 10 time 495 nodes 306501 nps 618898 pv e2e4 d7d5 e4e5 d5d4 g1f3 b8c6 f1b5 d8d5 c2c4 d5c5
info depth 11 score cp 29 time 1184 nodes 742958 nps 627410 pv e2e4 b8c6 b1c3 g8f6 g1f3 d7d5 e4e5 f6e4 f1b5 a7a6 b5c6
info depth 12 score cp 29 time 1435 nodes 906250 nps 631469 pv e2e4 b8c6 b1c3 g8f6 d2d4 d7d5 e4e5 f6e4 c3e4 d5e4 g1e2
 c8g4
info depth 13 score cp 35 time 2332 nodes 1481470 nps 635151 pv e2e4 b8c6 b1c3 g8f6 d2d4 d7d5 e4e5 f6e4 c3e4 d5e4 g1e2
 c8g4 c1e3
info depth 14 score cp 38 time 4219 nodes 2723830 nps 645467 pv e2e4 b8c6 d2d4 d7d5 e4d5 d8d5 g1f3 d5e4 f1e2 c8f5 b1c3
 e4c2 d1c2 f5c2
info depth 14 score cp 38 time 60000 nodes 36682755 nps 611369 pv e2e4 b8c6 d2d4 d7d5 e4d5 d8d5 g1f3 d5e4 f1e2 c8f5 b1c3
 e4c2 d1c2 f5c2
bestmove e2e4

info depth 1 score cp 54 time 0 nodes 26 nps 260000 pv b1c3
info depth 2 score cp 8 time 0 nodes 88 nps 178913 pv b1c3 g8f6
info depth 3 score cp 53 time 0 nodes 378 nps 405174 pv b1c3 b8c6 g1f3
info depth 4 score cp 8 time 3 nodes 1571 nps 476515 pv b1c3 b8c6 g1f3 g8f6
info depth 5 score cp 41 time 6 nodes 3258 nps 530351 pv b1c3 b8c6 g1f3 g8f6 e2e4
info depth 6 score cp 1 time 22 nodes 13016 nps 572434 pv b1c3 b8c6 g1f3 g8f6 e2e4 e7e5
info depth 7 score cp 18 time 33 nodes 19294 nps 579678 pv b1c3 b8c6 g1f3 g8f6 d2d4 d7d5 c1g5
info depth 8 score cp 11 time 114 nodes 67331 nps 588392 pv e2e4 b8c6 b1c3 g8f6
info depth 9 score cp 26 time 160 nodes 94592 nps 591189 pv e2e4 b8c6 b1c3 g8f6 g1f3 d7d5 e4e5 f6e4 c3e4
info depth 10 score cp 10 time 520 nodes 308014 nps 592079 pv e2e4 d7d5 e4e5 d5d4 g1f3 b8c6 f1b5 d8d5 c2c4 d5c5
info depth 11 score cp 29 time 1310 nodes 789780 nps 602608 pv e2e4 b8c6 b1c3 g8f6 g1f3 d7d5 e4e5 f6e4 f1b5 a7a6 b5c6
info depth 12 score cp 29 time 1591 nodes 964666 nps 606182 pv e2e4 b8c6 b1c3 g8f6 d2d4 d7d5 e4e5 f6e4 c3e4 d5e4 g1e2
 c8g4
info depth 13 score cp 35 time 2844 nodes 1733482 nps 609443 pv e2e4 b8c6 b1c3 g8f6 d2d4 d7d5 e4e5 f6e4 c3e4 d5e4 g1e2
 c8g4 c1e3
info depth 14 score cp 27 time 13669 nodes 8128351 nps 594654 pv e2e4 e7e5 b1c3 g8f6 g1f3 f8b4 a2a3 b4c3 d2c3 e8g8 f1d3
 d7d5 e1g1 d5e4
info depth 14 score cp 27 time 60000 nodes 35792898 nps 596540 pv e2e4 e7e5 b1c3 g8f6 g1f3 f8b4 a2a3 b4c3 d2c3 e8g8 f1d3
 d7d5 e1g1 d5e4
bestmove e2e4

LMR improvements
info depth 1 score cp 54 time 0 nodes 21 nps 196608 pv b1c3
info depth 2 score cp 8 time 0 nodes 82 nps 142651 pv b1c3 g8f6
info depth 3 score cp 53 time 1 nodes 275 nps 256775 pv b1c3 b8c6 g1f3
info depth 4 score cp 8 time 3 nodes 1219 nps 321644 pv b1c3 b8c6 g1f3 g8f6
info depth 5 score cp 41 time 6 nodes 2293 nps 375362 pv b1c3 b8c6 g1f3 g8f6 e2e4
info depth 6 score cp 1 time 16 nodes 6897 nps 428678 pv b1c3 b8c6 g1f3 g8f6 e2e4 e7e5
info depth 7 score cp 18 time 28 nodes 12917 nps 459816 pv b1c3 g8f6 g1f3 d7d5 d2d4 b8c6 c1g5
info depth 8 score cp 3 time 59 nodes 30915 nps 523416 pv e2e4 g8f6 b1c3 e7e6
info depth 9 score cp 26 time 126 nodes 65746 nps 519509 pv e2e4 b8c6 b1c3 g8f6
info depth 10 score cp 25 time 244 nodes 134603 nps 550526 pv e2e4 d7d5 e4d5 g8f6 g1f3 f6d5 d2d4
info depth 11 score cp 20 time 539 nodes 311258 nps 576836 pv e2e4 b8c6 b1c3 g8f6 d2d4 d7d5 f1b5 d5e4
info depth 12 score cp 27 time 1018 nodes 601283 nps 590314 pv e2e4 b8c6 d2d4 g8f6 b1c3 f6e4 c3e4 c6d4
info depth 13 score cp 28 time 2487 nodes 1484696 nps 596813 pv e2e4 b8c6 g1f3 g8f6 e4e5 f6g4 d2d4 g4f6
info depth 14 score cp 49 time 5101 nodes 3125994 nps 612752 pv e2e4 e7e5 b1c3 g8f6 g1f3 b8c6 d2d4 f8d6 d4e5 c6e5 c1f4
 e5f3
info depth 15 score cp 36 time 29248 nodes 17511637 nps 598711 pv e2e4 e7e5 b1c3 g8f6 g1f3 f8b4 f3e5 d8e7 e5d3 b4c3 d2c3
 e7e4 c1e3 g7g5 a2a3
bestmove e2e4
'''

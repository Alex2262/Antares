
"""
Antares V4 fully JIT compiled with Numba ~ 68k nps
Position class: contains board, castling ability, king position, en passant square
Board representation: 12x10 mailbox
Move representation: integer (28 bits used)

"""

import timeit

from evaluation import *
from move_generator import *
from position import *
from transposition import *

search_spec = [
    ("max_depth", nb.uint16),
    ("max_qdepth", nb.uint16),
    ("max_time", nb.double),        # seconds
    ("min_depth", nb.uint16),
    ("current_search_depth", nb.uint16),
    ("ply", nb.uint16),             # opposite of depth counter
    ("start_time", nb.uint64),
    ("node_count", nb.uint64),
    ("pv_table", nb.uint64[:, :]),  # implementation of pv and pv scoring comes from TSCP engine
    ("pv_length", nb.uint64[:]),
    ("follow_pv", nb.boolean),
    ("score_pv", nb.boolean),
    ("killer_moves", nb.uint64[:, :]),
    ("history_moves", nb.uint64[:, :]),
    ("transposition_table", NUMBA_HASH_TYPE[:]),
    ("stopped", nb.boolean)

]


@jitclass(spec=search_spec)
class Search:

    def __init__(self):
        # self.TRANSPOSITION_TABLE = {}
        # self.WIN_TABLE = {}

        self.max_depth = 15  # seems this number is multiplied by 2
        self.max_qdepth = 4
        self.min_depth = 2
        self.current_search_depth = 0
        self.ply = 0

        self.max_time = 10
        self.start_time = 0

        self.node_count = 0

        self.pv_table = np.zeros((self.max_depth, self.max_depth), dtype=np.uint64)
        self.pv_length = np.zeros(self.max_depth, dtype=np.uint64)
        self.follow_pv = False
        self.score_pv = False

        # Killer moves [id][ply]
        self.killer_moves = np.zeros((2, self.max_depth), dtype=np.uint64)
        # History moves [piece][square]
        self.history_moves = np.zeros((12, 64), dtype=np.uint64)

        self.transposition_table = np.zeros(MAX_HASH_SIZE, dtype=NUMBA_HASH_TYPE)

        self.stopped = False

    def reset(self):
        self.node_count = 0

        self.ply = 0

        self.pv_table = np.zeros((self.max_depth, self.max_depth), dtype=np.uint64)
        self.pv_length = np.zeros(self.max_depth, dtype=np.uint64)

        self.killer_moves = np.zeros((2, self.max_depth), dtype=np.uint64)
        self.history_moves = np.zeros((12, 64), dtype=np.uint64)
        self.transposition_table = np.zeros(MAX_HASH_SIZE, dtype=NUMBA_HASH_TYPE)
        
        self.follow_pv = False
        self.score_pv = False

        self.stopped = False

        # self.aspiration_window = 65  # in centi pawns


# Update the elapsed search time to determine when the search should be terminated.
@nb.njit
def update_search(engine):
    with nb.objmode(elapsed_time=nb.double):  # In a numba function use objmode to use regular python
        elapsed_time = timeit.default_timer() - engine.start_time
    if elapsed_time >= engine.max_time and engine.current_search_depth >= engine.min_depth:
        engine.stopped = True


@nb.njit
def enable_pv_scoring(engine, moves):
    engine.follow_pv = False

    if engine.pv_table[0][engine.ply] in moves:
        engine.score_pv = True
        engine.follow_pv = True


@nb.njit
def qsearch(engine, position, alpha, beta, depth):

    # Update the search progress every 1024 nodes
    if engine.node_count % 1024 == 0:
        update_search(engine)

    # Increase the node count
    engine.node_count += 1

    # The stand pat:
    # We assume that in a position there should always be a quiet evaluation,
    # and we return this if it is better than the captures.
    static_eval = evaluate(position)

    # Beta is the alpha (the best evaluation) of the previous node
    if static_eval >= beta:
        return beta
    if depth == 0:
        return static_eval

    # If our static evaluation has improved after the last move.
    alpha = max(alpha, static_eval)

    # Retrieving all pseudo legal captures as a list of [Move class]
    moves = get_scored_captures(get_pseudo_legal_captures(position))

    # Iterate through the noisy moves (captures) and search recursively with qsearch (quiescence search)
    for current_move_index in range(len(moves)):
        sort_next_move(moves, current_move_index)
        move = moves[current_move_index][0]

        # Make the capture
        attempt = make_capture(position, move)
        if not attempt:
            undo_capture(position, move)
            continue

        flip_position(position)

        return_eval = -qsearch(engine, position, -beta, -alpha, depth - 1)

        flip_position(position)
        undo_capture(position, move)

        # beta cutoff
        if return_eval >= beta:
            return beta

        alpha = max(alpha, return_eval)

    return alpha


@nb.njit
def negamax(engine, position, alpha, beta, depth):

    # position.hash_key = compute_hash(position)

    # Initialize PV length
    engine.pv_length[engine.ply] = engine.ply

    # Start quiescence search at the end of regular negamax search to counter the horizon effect.
    if depth == 0:
        return qsearch(engine, position, alpha, beta, engine.max_qdepth)

    # Increase node count after checking for terminal nodes since that would be counting double
    # nodes with quiescent search
    engine.node_count += 1

    # Terminate search when a condition has been reached, usually a time limit.
    if engine.stopped:
        return 0

    # Get a value from probe_tt_entry that will correspond to either returning a score immediately
    # or returning no hash entry, or returning move int to sort
    tt_value = probe_tt_entry(engine, position, alpha, beta, depth)
    tt_move = NO_MOVE

    # A score was given to return
    if tt_value < NO_HASH_ENTRY:
        return tt_value

    # Use a tt entry move to sort moves
    elif tt_value > USE_HASH_MOVE:
        tt_move = tt_value - USE_HASH_MOVE

    # Using a variable to record the hash flag
    tt_hash_flag = HASH_FLAG_ALPHA

    # Used at the end of the negamax function to determine checkmates, stalemates, etc.
    legal_moves = 0
    in_check = is_attacked(position, position.own_king_position)

    # Check extension
    if in_check:
        depth += 1

    # Saving EP information ahead for null moves
    current_ep = position.ep_square

    # Null move pruning
    # We give the opponent an extra move and if they are not able to make their position
    # any better, then our position is too good, and we don't need to search any deeper.
    if depth >= 3 and not in_check and engine.ply:

        # Make the null move (flipping the position and clearing the ep square)
        flip_position(position)
        make_null_move(position)

        engine.ply += 1

        # We will reduce the depth since the opponent gets two moves in a row to improve their position
        return_eval = -negamax(engine, position, -beta, -beta + 1, depth - 1 - NULL_MOVE_REDUCTION)
        engine.ply -= 1

        # Undoing the null move (flipping the position back and setting ep square to current ep square)
        flip_position(position)
        undo_null_move(position, current_ep)

        if return_eval >= beta:
            return beta

    # Retrieving the pseudo legal moves in the current position as a list of integers
    moves = get_pseudo_legal_moves(position)

    # Check if we are following the PV line
    if engine.follow_pv:
        enable_pv_scoring(engine, moves)

    # Score the moves but do not sort them yet
    moves = get_scored_moves(engine, moves, tt_move)

    # Saving information from the current position without reference to successfully undo moves.
    current_own_castle_ability = np.full(2, True)
    current_own_castle_ability[0] = position.own_castle_ability[0]
    current_own_castle_ability[1] = position.own_castle_ability[1]
    current_opp_castle_ability = np.full(2, True)
    current_opp_castle_ability[0] = position.opp_castle_ability[0]
    current_opp_castle_ability[1] = position.opp_castle_ability[1]

    # Best move to save for TT
    best_move = NO_MOVE

    # Iterate through moves and recursively search with Negamax
    for current_move_index in range(len(moves)):

        # Sort the next move. If an early move causes a cutoff then we have saved time
        # by only sorting one or a few moves rather than the whole list.
        sort_next_move(moves, current_move_index)
        move = moves[current_move_index][0]

        # Make the move
        attempt = make_move(position, move)

        # The move put us in check, therefore it was not legal, and we must disregard it
        if not attempt:
            undo_move(position, move, current_own_castle_ability, current_opp_castle_ability, current_ep)
            continue

        # We proceed since the move is legal
        # Flip the position for the opposing player
        flip_position(position)

        # Increase legal moves and increase ply
        legal_moves += 1
        engine.ply += 1

        # Normal search for the first move
        if legal_moves == 0:
            return_eval = -negamax(engine, position, -beta, -alpha, depth - 1)

        # Different Reduction Methods
        else:
            # Late Move Reductions (LMR)
            # Conditions to consider a LMR
            if legal_moves >= FULL_DEPTH_MOVES and depth >= REDUCTION_LIMIT and\
               not in_check and get_move_type(move) == 0 and not get_is_capture(move):
                return_eval = -negamax(engine, position, -alpha - 1, -alpha, depth - 2)
            else:
                return_eval = alpha + 1

            # Principle Variation Search
            # If the 'pv node' has possibly been 'found', then we search with a narrower window
            if return_eval > alpha:
                # Search with a narrower window
                return_eval = -negamax(engine, position, -alpha - 1, -alpha, depth - 1)
                # If the narrower window search has failed us we must search with a full window again
                if alpha < return_eval < beta:
                    # Research at a full depth
                    return_eval = -negamax(engine, position, -beta, -alpha, depth - 1)

        # Decrease the ply
        engine.ply -= 1

        # Flip the position back to us and undo the move
        flip_position(position)
        undo_move(position, move, current_own_castle_ability, current_opp_castle_ability, current_ep)

        # The move is better than other moves searched
        if return_eval > alpha:

            alpha = return_eval
            best_move = move

            # Exact flag reached
            tt_hash_flag = HASH_FLAG_EXACT

            # Store history moves
            if not get_is_capture(move):
                engine.history_moves[get_selected(move)][MAILBOX_TO_STANDARD[get_to_square(move)]] += depth * depth

            # Write pv move
            engine.pv_table[engine.ply][engine.ply] = move
            for next_ply in range(engine.ply + 1, engine.pv_length[engine.ply + 1]):
                engine.pv_table[engine.ply][next_ply] = engine.pv_table[engine.ply + 1][next_ply]

            engine.pv_length[engine.ply] = engine.pv_length[engine.ply + 1]

            # Alpha - Beta cutoff. This is a 'cut node' and we have failed high
            if alpha >= beta:

                # Store killer moves
                if not get_is_capture(move):
                    engine.killer_moves[1][engine.ply] = engine.killer_moves[0][engine.ply]
                    engine.killer_moves[0][engine.ply] = move

                record_tt_entry(engine, position, beta, HASH_FLAG_BETA, move, depth)
                return beta

    # Stalemate: No legal moves and king is not in check
    if legal_moves == 0 and not in_check:
        return 0
    # Checkmate: No legal moves and king is in check
    elif legal_moves == 0 and in_check:
        return -MATE_SCORE - depth

    record_tt_entry(engine, position, alpha, tt_hash_flag, best_move, depth)

    # We return our best score possible. This is an 'All node' and we have failed low
    return alpha


# An iterative search approach to negamax
def iterative_search(engine, position):

    # Reset engine variables
    engine.stopped = False
    engine.start_time = timeit.default_timer()
    engine.reset()

    # Total nodes
    node_sum = 0

    # Prepare window for negamax search
    alpha = -1000000
    beta = 1000000

    running_depth = 1

    best_pv = ""
    best_score = 0

    while running_depth <= engine.max_depth:
        # Reset engine variables
        engine.node_count = 0
        engine.current_search_depth = running_depth

        # Enable following pv
        engine.follow_pv = True

        # Negamax search
        returned = negamax(engine, position, alpha, beta, running_depth)

        # Reset the window
        if returned <= alpha or returned >= beta:
            alpha = -INF
            beta = INF
            continue

        # Adjust aspiration window
        alpha = returned - ASPIRATION_VAL
        beta = returned + ASPIRATION_VAL

        # Add to total node counts
        node_sum += engine.node_count
        # Obtain principle variation line and print it
        pv_line = []
        for c in range(engine.pv_length[0]):
            pv_line.append(get_uci_from_move(position, engine.pv_table[0][c]))
            position.side ^= 1

        best_pv = pv_line if not engine.stopped else best_pv
        best_score = returned if not engine.stopped else best_score

        if not engine.stopped:
            print(f"info depth {running_depth} score cp {returned} "
                  f"time {int((timeit.default_timer() - engine.start_time) * 1000)} nodes {engine.node_count} "
                  f"nps {int(node_sum / (timeit.default_timer() - engine.start_time))} "
                  f"pv {' '.join(best_pv)}")

            running_depth += 1
        else:
            print(f"info depth {running_depth-1} score cp {best_score} "
                  f"time {int((timeit.default_timer() - engine.start_time) * 1000)} nodes {engine.node_count} "
                  f"nps {int(node_sum / (timeit.default_timer() - engine.start_time))} "
                  f"pv {' '.join(best_pv)}")
            print(f"bestmove {best_pv[0]}")
            break

    return


def compile_engine(engine, position):
    start_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    position.parse_fen(start_fen)

    alpha = -1000000
    beta = 1000000
    returned = negamax(engine, position, alpha, beta, 2)

    return returned


''' No ordering
1 0.1986857079999993 0.58 1 1 5 5
2 0.19898225000000025 0.0 21 22 94 99
3 0.20294883300000066 0.58 277 299 796 895
4 0.2540480830000007 0.0 4213 4512 15121 16016
5 0.7658198330000001 0.41 39801 44313 123560 139576
6 4.629144374999999 0.0 296778 341091 1317306 1456882
7 43.06958825 0.3 2639583 2980674 11574655 13031537
UNFINISHED 60.017578625 1144000 4124674 5518826 18550363
'''

''' Move Ordering!!
1 0.2423271249999992 97 76 0.58 1 1 1 1
2 0.24245779199999973 97 76 0.0 21 22 20 21
3 0.24329862499999955 97 76 0.58 60 82 20 41
4 0.2465868750000002 97 76 0.0 524 606 556 597
5 0.2638864590000001 97 76 0.41 1424 2030 910 1507
6 0.38953879199999975 97 76 0.0 14077 16107 30927 32434
7 1.2136407499999997 85 65 0.3 58633 74740 109453 141887
8 8.362156709 97 76 0.0 445415 520155 1999461 2141348
UNFINISHED 60.012888958999994 1906000 2426155 13584540 15725888
'''

''' PV sorting!
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
'''

'''Principle Variation Search!!
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
'''


''' Killer and History moves!
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
'''


''' Late move reductions!!!
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
'''

''' Null move pruning!!! -- probably has even more effect in tactical positions
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
'''

''' Aspiration Windows !! -- Also included Q nodes in node count since that is the standard
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
'''

''' History moves fixed to be HH += depth * depth instead of HH += depth
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
'''

''' Quiescence search terminal nodes counted since this is the default
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
'''


''' LMR full depth moves changed to 3
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
info depth 1 score cp 94 time 28 nodes 26 nps 920 pv b1c3
info depth 2 score cp 50 time 28 nodes 86 nps 3949 pv g8f6 b1c3
info depth 3 score cp 94 time 28 nodes 211 nps 11280 pv g8f6 b1c3 b8c6
info depth 4 score cp 50 time 32 nodes 1564 nps 58852 pv b1c3 g8f6 g1f3 b8c6
info depth 5 score cp 91 time 35 nodes 3196 nps 141237 pv b1c3 b8c6 g1f3 g8f6 d2d4
info depth 6 score cp 51 time 80 nodes 34879 nps 497449 pv b8c6 g1f3 e7e5 b1c3 g8e7 e2e4
info depth 7 score cp 79 time 113 nodes 27991 nps 600807 pv b8c6 b1c3 g8f6 e2e4 e7e5 g1e2 d7d6
info depth 8 score cp 60 time 431 nodes 261622 nps 764630 pv b1c3 g8f6 e2e4 d7d5 f1d3 d5e4 c3e4 b8d7
info depth 9 score cp 54 time 861 nodes 360442 nps 801188 pv b1c3 g8f6 e2e4 b8c6 d2d4 d7d5 e4d5 f6d5 g1e2
info depth 10 score cp 64 time 1862 nodes 802543 nps 801219 pv b8c6 b1c3 g8f6 d2d3 e7e5 g1f3 f8c5 e2e4 c6d4 c3d5
info depth 11 score cp 64 time 10614 nodes 7311427 nps 829402 pv g8f6 d2d3 b8c6 e2e4 e7e5 b1c3 d7d6 g1e2 c8g4 h2h3 g4e6
info depth 12 score cp 65 time 54082 nodes 37513705 nps 856432 pv e2e4 e7e6 g1f3 b8c6 b1c3 g8f6 d2d3 d7d5 e4e5 f6d7 
    c1g5 f8e7
info depth 12 score cp 65 time 60001 nodes 1 nps 771945 pv e2e4 e7e6 g1f3 b8c6 b1c3 g8f6 d2d3 d7d5 e4e5 f6d7 c1g5 f8e7
bestmove e2e4

info depth 12 score cp 65 time 1200001 nodes 1048079573 nps 911996 pv e2e4 e7e6 g1f3 b8c6 b1c3 g8f6 d2d3 d7d5 e4e5 f6d7 
    c1g5 f8e7
    
bestmove e2e4


HUGE BUG FIXED. Beta cutoffs were returning alpha instead of beta.
No changes in nodes though?

'''
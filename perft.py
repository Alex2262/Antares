
import numba as nb
import numpy as np
import timeit
from position import *
from move_generator import *


@nb.njit(cache=False)
def debug_perft(position, depth):

    if depth == 0:
        return 1, 0, 0, 0, 0, 0

    moves = get_pseudo_legal_moves(position)

    # -----
    current_ep = position.ep_square
    current_own_castle_ability = np.full(2, True)
    current_own_castle_ability[0] = position.own_castle_ability[0]
    current_own_castle_ability[1] = position.own_castle_ability[1]
    current_opp_castle_ability = np.full(2, True)
    current_opp_castle_ability[0] = position.opp_castle_ability[0]
    current_opp_castle_ability[1] = position.opp_castle_ability[1]

    amt = 0
    capture_amt = 0
    ep_amt = 0
    check_amt = 0
    promotion_amt = 0
    castle_amt = 0

    for move in moves:
        attempt = make_move(position, move)

        if not attempt:
            undo_move(position, move, current_own_castle_ability, current_opp_castle_ability, current_ep)
            continue

        flip_position(position)

        if depth == 1:

            move_type = get_move_type(move)

            if get_is_capture(move):
                capture_amt += 1
            elif move_type == 1:
                capture_amt += 1
                ep_amt += 1
            elif move_type == 3:
                promotion_amt += 1
            elif move_type == 2:
                castle_amt += 1

            if is_attacked(position, position.own_king_position):
                check_amt += 1

        returned = debug_perft(position, depth - 1)

        amt += returned[0]
        capture_amt += returned[1]
        ep_amt += returned[2]
        check_amt += returned[3]
        promotion_amt += returned[4]
        castle_amt += returned[5]

        flip_position(position)
        undo_move(position, move, current_own_castle_ability, current_opp_castle_ability, current_ep)

    return amt, capture_amt, ep_amt, check_amt, promotion_amt, castle_amt


@nb.njit(cache=False)
def fast_perft(position, depth):
    if depth == 0:
        return 1

    moves = get_pseudo_legal_moves(position)

    # -----
    current_ep = position.ep_square
    current_own_castle_ability = np.full(2, True)
    current_own_castle_ability[0] = position.own_castle_ability[0]
    current_own_castle_ability[1] = position.own_castle_ability[1]
    current_opp_castle_ability = np.full(2, True)
    current_opp_castle_ability[0] = position.opp_castle_ability[0]
    current_opp_castle_ability[1] = position.opp_castle_ability[1]

    amt = 0

    for move in moves:

        attempt = make_move(position, move)

        if not attempt:
            undo_move(position, move, current_own_castle_ability, current_opp_castle_ability, current_ep)
            continue

        flip_position(position)

        amt += fast_perft(position, depth - 1)

        flip_position(position)
        undo_move(position, move, current_own_castle_ability, current_opp_castle_ability, current_ep)

    return amt


@nb.njit(cache=False)
def uci_perft(position, depth):

    with nb.objmode(start_time=nb.double):
        start_time = timeit.default_timer()

    # In case someone decides to run perft 0?
    if depth == 0:
        return 1

    moves = get_pseudo_legal_moves(position)
    total_amt = 0

    # -----
    current_ep = position.ep_square
    current_own_castle_ability = np.full(2, True)
    current_own_castle_ability[0] = position.own_castle_ability[0]
    current_own_castle_ability[1] = position.own_castle_ability[1]
    current_opp_castle_ability = np.full(2, True)
    current_opp_castle_ability[0] = position.opp_castle_ability[0]
    current_opp_castle_ability[1] = position.opp_castle_ability[1]

    for move in moves:

        attempt = make_move(position, move)

        if not attempt:
            undo_move(position, move, current_own_castle_ability, current_opp_castle_ability, current_ep)
            continue

        flip_position(position)

        amt = fast_perft(position, depth - 1)
        total_amt += amt

        flip_position(position)
        undo_move(position, move, current_own_castle_ability, current_opp_castle_ability, current_ep)

        print("Move " + get_uci_from_move(position, move) + ": " + str(amt))

    with nb.objmode:    # (end_time=nb.float64)
        end_time = timeit.default_timer()

        print("nodes searched: " + str(total_amt))
        print("perft speed: " + str(int(total_amt / (end_time - start_time)) / 1000) + "kn/s")
        print("total time: " + str(end_time - start_time))

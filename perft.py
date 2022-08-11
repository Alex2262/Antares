
import time

from move_generator import *
from position import *


@nb.njit(cache=False)
def debug_perft(position, depth):

    if depth == 0:
        return 1, 0, 0, 0, 0, 0

    moves = get_pseudo_legal_moves(position)

    # -----
    current_ep = position.ep_square
    current_castle_ability_bits = position.castle_ability_bits
    current_hash_key = position.hash_key

    amt = 0
    capture_amt = 0
    ep_amt = 0
    check_amt = 0
    promotion_amt = 0
    castle_amt = 0

    for move in moves:

        attempt = make_move(position, move)

        if not attempt:
            undo_move(position, move, current_ep, current_castle_ability_bits, current_hash_key)
            continue

        position.side ^= 1

        if depth == 1:

            move_type = get_move_type(move)

            if get_is_capture(move):
                capture_amt += 1
            elif move_type == MOVE_TYPE_EP:
                capture_amt += 1
                ep_amt += 1
            elif move_type == MOVE_TYPE_PROMOTION:
                promotion_amt += 1
            elif move_type == MOVE_TYPE_CASTLE:
                castle_amt += 1

            if is_attacked(position, position.king_positions[position.side]):
                check_amt += 1

        returned = debug_perft(position, depth - 1)

        amt += returned[0]
        capture_amt += returned[1]
        ep_amt += returned[2]
        check_amt += returned[3]
        promotion_amt += returned[4]
        castle_amt += returned[5]

        position.side ^= 1
        undo_move(position, move, current_ep, current_castle_ability_bits, current_hash_key)

    return amt, capture_amt, ep_amt, check_amt, promotion_amt, castle_amt


@nb.njit(cache=False)
def fast_perft(position, depth):
    if depth == 0:
        return 1

    moves = get_pseudo_legal_moves(position)

    # -----
    current_ep = position.ep_square
    current_castle_ability_bits = position.castle_ability_bits
    current_hash_key = position.hash_key

    amt = 0

    for move in moves:

        attempt = make_move(position, move)

        if not attempt:
            undo_move(position, move, current_ep, current_castle_ability_bits, current_hash_key)
            continue

        position.side ^= 1
        amt += fast_perft(position, depth - 1)
        position.side ^= 1

        undo_move(position, move, current_ep, current_castle_ability_bits, current_hash_key)

    return amt


@nb.njit(cache=False)
def uci_perft(position, depth):

    with nb.objmode(start_time=nb.double):
        start_time = time.time()

    # In case someone decides to run perft 0?
    if depth == 0:
        return 1

    moves = get_pseudo_legal_moves(position)
    total_amt = 0

    # -----
    current_ep = position.ep_square
    current_castle_ability_bits = position.castle_ability_bits
    current_hash_key = position.hash_key

    for move in moves:

        attempt = make_move(position, move)

        if not attempt:
            undo_move(position, move, current_ep, current_castle_ability_bits, current_hash_key)
            continue

        position.side ^= 1
        amt = fast_perft(position, depth - 1)
        total_amt += amt

        position.side ^= 1
        undo_move(position, move, current_ep, current_castle_ability_bits, current_hash_key)

        print("Move " + get_uci_from_move(move) + ": " + str(amt))

    with nb.objmode:
        end_time = time.time()

        print("nodes searched: " + str(total_amt))
        print("perft speed: " + str(int(total_amt / (end_time - start_time)) / 1000) + "kn/s")
        print("total time: " + str(end_time - start_time))


'''
incheck njit
(1, 0, 0, 0, 0) 3.624999999729539e-06
(20, 0, 0, 0, 0) 0.5131300830000001
(400, 0, 0, 0, 0) 0.010259709000001394
(8902, 34, 0, 12, 0) 0.25145966700000066

3: (1, 0, 0, 0, 0) 3.9590000000533365e-06
3: (20, 0, 0, 0, 0) 0.9135626250000001
3: (400, 0, 0, 0, 0) 0.02379966599999994
3: (8902, 34, 0, 12, 0) 0.5183446670000003

Makemove njit
(1, 0, 0, 0, 0) 1.3330000001587905e-06
(20, 0, 0, 0, 0) 0.8349006249999995
(400, 0, 0, 0, 0) 0.011415915999999804
(8902, 34, 0, 12, 0) 0.26993779199999945

3: (1, 0, 0, 0, 0) 3.250000000010189e-06
3: (20, 0, 0, 0, 0) 1.092408416
3: (400, 0, 0, 0, 0) 0.4462142920000003
3: (8902, 34, 0, 12, 0) 0.2711170829999996

Undomove njit
(1, 0, 0, 0, 0) 1.708000000100185e-06
(20, 0, 0, 0, 0) 1.0710964580000004
(400, 0, 0, 0, 0) 0.01103325000000055
(8902, 34, 0, 12, 0) 0.23741866700000003

3: (1, 0, 0, 0, 0) 2.8749999999577724e-06
3: (20, 0, 0, 0, 0) 1.192687125
3: (400, 0, 0, 0, 0) 0.47335604200000003
3: (8902, 34, 0, 12, 0) 0.27404020900000026

Pseudo Legal Move
(1, 0, 0, 0, 0) 9.58000000217396e-07
(20, 0, 0, 0, 0) 1.3550063749999994
(400, 0, 0, 0, 0) 0.002123750000000868
(8902, 34, 0, 12, 0) 0.04826562500000087

3: (1, 0, 0, 0, 0) 3.749999999969056e-06
3: (20, 0, 0, 0, 0) 1.6368319580000001
3: (400, 0, 0, 0, 0) 0.782766375
3: (8902, 34, 0, 12, 0) 0.23104925000000032

Perft
(1, 0, 0, 0, 0) 1.6898107920000003
(20, 0, 0, 0, 0) 0.0004498330000011208
(400, 0, 0, 0, 0) 0.0002648749999991651
(8902, 34, 0, 12, 0) 0.005955042000000077

3: (1, 0, 0, 0, 0) 2.778665666
3: (20, 0, 0, 0, 0) 0.00040354200000036755
3: (400, 0, 0, 0, 0) 0.00028866599999988196
3: (8902, 34, 0, 12, 0) 0.0043427080000002505

--
(1, 0, 0, 0, 0) 1.615739875000001
(20, 0, 0, 0, 0) 0.0004347500000001503
(400, 0, 0, 0, 0) 0.0002569169999997456
(8902, 34, 0, 12, 0) 0.005752792000000895
(197281, 1576, 0, 469, 461) 0.12309824999999996
(4865609, 82719, 258, 27351, 15375) 3.134478208000001

3: (1, 0, 0, 0, 0) 2.963718333
3: (20, 0, 0, 0, 0) 0.0004398749999996454
3: (400, 0, 0, 0, 0) 0.00031437499999986684
3: (8902, 34, 0, 12, 0) 0.004687374999999605
3: (197281, 1576, 0, 469, 461) 0.14667049999999993
3: (4865609, 82719, 258, 27351, 15375) 2.412793375

Removing flip_position
NODES 20 CAPTURES 0 EP 0 CHECKS 0 PROMOTIONS 0 CASTLES 0 TIME 2401
NODES 400 CAPTURES 0 EP 0 CHECKS 0 PROMOTIONS 0 CASTLES 0 TIME 0
NODES 8902 CAPTURES 34 EP 0 CHECKS 12 PROMOTIONS 0 CASTLES 0 TIME 2
NODES 197281 CAPTURES 1576 EP 0 CHECKS 469 PROMOTIONS 0 CASTLES 0 TIME 51
NODES 4865609 CAPTURES 82719 EP 258 CHECKS 27351 PROMOTIONS 0 CASTLES 0 TIME 1226

Piece lists
NODES 20 CAPTURES 0 EP 0 CHECKS 0 PROMOTIONS 0 CASTLES 0 TIME 2845
NODES 400 CAPTURES 0 EP 0 CHECKS 0 PROMOTIONS 0 CASTLES 0 TIME 0
NODES 8902 CAPTURES 34 EP 0 CHECKS 12 PROMOTIONS 0 CASTLES 0 TIME 2
NODES 197281 CAPTURES 1576 EP 0 CHECKS 469 PROMOTIONS 0 CASTLES 0 TIME 53
NODES 4865609 CAPTURES 82719 EP 258 CHECKS 27351 PROMOTIONS 0 CASTLES 0 TIME 1377
'''
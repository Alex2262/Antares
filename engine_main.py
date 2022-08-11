
"""
This file is solely for testing purposes.
The main file which handles uci is main.py
"""

from cache_clearer import kill_numba_cache
from perft import *
from search import iterative_search
from search_class import Search, init_search, SearchStruct_set_max_time, SearchStruct_set_max_depth
from position_class import init_position


def main():

    start_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    # kiwipete_fen = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - "
    # test_3 = "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - "
    # test_4 = "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1"
    # test_5 = "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8"
    # new_fen = "8/8/7P/1r3pR1/4k3/3p4/6PK/8 w - - 0 1"

    position_fen = start_fen

    print("compiling...")
    start = time.time()

    main_position = init_position()

    parse_fen(main_position, position_fen)

    main_engine = init_search()

    print(make_readable_board(main_position))

    SearchStruct_set_max_time(main_engine, 20000)
    SearchStruct_set_max_depth(main_engine, 3)
    iterative_search(main_engine, main_position, True)

    print("\n----- ----- ----- ----- ----- -----\n")
    parse_fen(main_position, position_fen)

    print(time.time() - start)

    SearchStruct_set_max_time(main_engine, 60000)
    SearchStruct_set_max_depth(main_engine, 64)
    iterative_search(main_engine, main_position, False)

    '''for i in range(1, 6):
        start = time.time()
        returned = debug_perft(main_position, i)
        print(f"NODES {returned[0]} CAPTURES {returned[1]} EP {returned[2]} CHECKS {returned[3]} "
              f"PROMOTIONS {returned[4]} CASTLES {returned[5]} "
              f"TIME {int((time.time() - start) * 1000)}")'''

    # print(debug_perft(main_position, 5), time.time() - start)
    # uci_perft(main_position, 5)

    '''print(PIECE_HASH_KEYS)
    print(SIDE_HASH_KEY)
    print(EP_HASH_KEYS)
    print(CASTLE_HASH_KEYS)'''


def clear_cache():
    kill_numba_cache()


if __name__ == "__main__":
    main()
    # clear_cache()

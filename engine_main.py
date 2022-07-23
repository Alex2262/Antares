
"""
This file is solely for testing purposes.
The main file which handles uci is main.py
"""

from cache_clearer import kill_numba_cache
from perft import *
from search import Search, iterative_search


def main():

    start_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    # kiwipete_fen = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - "
    # test_3 = "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - "
    # test_4 = "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1"
    # test_5 = "r2q1rk1/ppp2ppp/2n1bn2/2b1p3/3pP3/3P1NPP/PPP1NPB1/R1BQ1RK1 b - - 0 1"

    print("compiling...")
    start = timeit.default_timer()
    main_position = Position()

    main_position.parse_fen(start_fen)

    main_engine = Search()
    print(timeit.default_timer() - start)

    print(main_position.make_readable_board())

    main_engine.max_time = 20
    main_engine.max_depth = 3
    iterative_search(main_engine, main_position)

    print("\n----- ----- ----- ----- ----- -----\n")

    main_position.parse_fen(start_fen)

    print(main_position.make_readable_board())
    main_engine.max_time = 60
    main_engine.max_depth = 15
    iterative_search(main_engine, main_position)

    '''for i in range(1, 2):
        start = timeit.default_timer()
        print(debug_perft(main_position, i), timeit.default_timer() - start)'''

    # print(debug_perft(main_position, 5), timeit.default_timer() - start)
    # uci_perft(main_position, 6)

    '''print(PIECE_HASH_KEYS)
    print(SIDE_HASH_KEY)
    print(EP_HASH_KEYS)
    print(CASTLE_HASH_KEYS)'''


def clear_cache():
    kill_numba_cache()


if __name__ == "__main__":
    main()
    # clear_cache()

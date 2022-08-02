
"""
UCI Handler
"""

import sys
import threading

from cache_clearer import kill_numba_cache
from move import get_move_from_uci
from position import make_move, parse_fen
from position_class import Position
from search import iterative_search, compile_engine
from search_class import Search


def parse_go(msg, engine, turn):
    """parse 'go' uci command"""

    d = engine.max_depth
    t = engine.max_time

    _, *params = msg.split()

    for p, v in zip(*2 * (iter(params),)):
        # print(p, v)
        if p == "depth":
            d = int(v)
        elif p == "movetime":
            t = int(v) / 2 / 1000
        elif p == "nodes":
            # n = int(v)
            pass
        elif p == "wtime":
            if turn == "w":
                t = int(v) / 20 / 1000
        elif p == "btime":
            if turn == "b":
                t = int(v) / 20 / 1000

    engine.max_time = t
    engine.max_depth = d


def main():
    """
    The main input/output loop.
    This implements a slice of the UCI protocol.
    """

    start_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    turn = "w"

    main_position = Position()
    main_engine = Search()

    compile_thread = threading.Thread(target=compile_engine, args=(main_engine, main_position))
    compile_thread.start()

    while True:
        msg = input().strip()
        print(f">>> {msg}", file=sys.stderr)

        tokens = msg.split()

        if msg == "quit":
            break

        elif msg == "uci" or msg.startswith("uciok"):
            print("id name AntaresPy2.22")
            print("id author Alexander_Tian")
            print("uciok")
            continue

        elif msg == "isready":
            compile_thread.join()
            print("readyok")
            continue

        elif msg == "ucinewgame":
            parse_fen(main_position, start_fen)
            turn = "w"

        elif msg.startswith("position"):
            if len(tokens) < 2:
                continue

            if tokens[1] == "startpos":
                parse_fen(main_position, start_fen)
                turn = "w"
                next_idx = 2

            elif tokens[1] == "fen":
                fen = " ".join(tokens[2:8])
                parse_fen(main_position, start_fen)
                turn = fen.strip().split()[1]
                next_idx = 8

            else:
                continue

            if len(tokens) <= next_idx or tokens[next_idx] != "moves":
                continue

            for move in tokens[(next_idx + 1):]:
                formatted_move = get_move_from_uci(main_position, move)
                make_move(main_position, formatted_move)

                main_position.side ^= 1
                turn = "w" if turn == "b" else "b"

        if msg.startswith("go"):
            parse_go(msg, main_engine, turn)
            iterative_search(main_engine, main_position, False)
            continue

    sys.exit()


def clear_cache():
    kill_numba_cache()


if __name__ == "__main__":
    main()
    # clear_cache()

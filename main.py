
"""
UCI Handler
"""

import sys
import threading

from cache_clearer import kill_numba_cache
from move import get_move_from_uci, get_is_capture
from position import make_move, parse_fen, is_attacked
from position_class import Position
from search import iterative_search, compile_engine, new_game
from search_class import Search
from utilities import NO_MOVE


def time_handler(engine, position, last_move, time, inc, movetime, movestogo):
    rate = 20
    if is_attacked(position, position.king_positions[position.side]):
        rate -= 3
    if get_is_capture(last_move):
        rate -= 1.5

    if movetime > 0:
        time_amt = movetime * 0.9
    elif inc > 0:
        if time < inc:
            time_amt = time / (rate / 10)
        else:
            time_amt = max(0.8 * inc + (time - inc * rate) / (rate*2), time / (rate*4))
    elif movestogo > 0:
        time_amt = (time * 0.8 / movestogo) * (20 / rate)
        if time_amt > time * 0.8:
            time_amt = time * 0.85
    elif time > 0:
        time_amt = time / rate
    else:
        time_amt = engine.max_time

    engine.max_time = int(time_amt)


def parse_go(engine, position, msg, last_move):
    """parse 'go' uci command"""

    d = engine.max_depth

    _, *params = msg.split()

    wtime = 0
    btime = 0
    winc = 0
    binc = 0
    movetime = 0
    movestogo = 0

    for p, v in zip(*2 * (iter(params),)):
        # print(p, v)
        if p == "depth":
            d = int(v)
        elif p == "nodes":
            # n = int(v)
            pass
        elif p == "movetime":
            movetime = int(v)
        elif p == "wtime":
            wtime = int(v)
        elif p == "btime":
            btime = int(v)
        elif p == "winc":
            winc = int(v)
        elif p == "binc":
            binc = int(v)
        elif p == "movestogo":
            movestogo = int(v)

    if position.side == 0:
        time = wtime
        inc = winc
    else:
        time = btime
        inc = binc

    time_handler(engine, position, last_move, time, inc, movetime, movestogo)
    engine.max_depth = d


def main():
    """
    The main input/output loop.
    This implements a slice of the UCI protocol.
    """

    start_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

    main_position = Position()
    main_engine = Search()

    compile_thread = threading.Thread(target=compile_engine, args=(main_engine, main_position))
    compile_thread.start()
    last_move = NO_MOVE

    while True:
        msg = input().strip()
        print(f">>> {msg}", file=sys.stderr)

        tokens = msg.split()

        if msg == "quit":
            break

        elif msg == "uci" or msg.startswith("uciok"):
            print("id name AntaresPy2.26")
            print("id author Alexander_Tian")
            print("uciok")
            continue

        elif msg == "isready":
            compile_thread.join()
            print("readyok")
            continue

        elif msg == "ucinewgame":
            parse_fen(main_position, start_fen)
            new_game(main_engine)
            last_move = NO_MOVE

        elif msg.startswith("position"):
            if len(tokens) < 2:
                continue

            if tokens[1] == "startpos":
                parse_fen(main_position, start_fen)
                next_idx = 2

            elif tokens[1] == "fen":
                fen = " ".join(tokens[2:8])
                parse_fen(main_position, fen)
                next_idx = 8

            else:
                continue

            if len(tokens) <= next_idx or tokens[next_idx] != "moves":
                continue

            main_engine.repetition_index = 0
            for move in tokens[(next_idx + 1):]:
                formatted_move = get_move_from_uci(main_position, move)
                last_move = formatted_move
                make_move(main_position, formatted_move)

                main_engine.repetition_index += 1
                main_engine.repetition_table[main_engine.repetition_index] = main_position.hash_key

                main_position.side ^= 1

        if msg.startswith("go"):
            parse_go(main_engine, main_position, msg, last_move)
            iterative_search(main_engine, main_position, False)
            continue

    sys.exit()


def clear_cache():
    kill_numba_cache()


if __name__ == "__main__":
    main()
    # clear_cache()
